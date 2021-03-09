import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

NUM_SEGMENTS = 22
HP_REG_FACTOR = 0
TV_REG_FACTOR = 0
CONTRAST_REG_FACTOR = 0


class ActionAttentionLSTM(nn.Module):

    def __init__(self, hidden_size: int, output_size: int):
        super().__init__()

        # Attention
        self.att_feature_w = nn.Linear(2048, 1, bias=False)
        self.att_hidden_w = nn.Linear(hidden_size, 1, bias=False)
        self.fc_out = nn.Linear(hidden_size, output_size)

        self.c0_conv = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.h0_conv = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.mask_conv = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),  # (bs*22, 1, 7, 7)
        )

        self.conv_lstm_cell = ConvLSTMCell(input_size=(7, 7),
                                           input_dim=2048,
                                           hidden_dim=512,
                                           kernel_size=(3, 3),
                                           bias=True)

        self.dropout_2d = nn.Dropout2d(p=0.2)

    def get_start_states(self, input_x):
        h0 = self.h0_conv(torch.mean(input_x, dim=1))
        c0 = self.c0_conv(torch.mean(input_x, dim=1))
        return h0, c0

    def temporal_attention_layer(self, features, hiddens):
        """
        : param features: (batch_size, 2048, 7, 7)
        : param hiddens: (batch_size, hidden_dim)
        :return:
        """

        # [30 x 2048 x 7 x 7]
        features_tmp = torch.mean(torch.mean(features, dim=3), dim=2)

        # [30 x 512 x 7 x 7]
        hiddens_tmp = torch.mean(torch.mean(hiddens, dim=3), dim=2)

        att_fea = self.att_feature_w(features_tmp)
        att_h = self.att_hidden_w(hiddens_tmp)

        return att_fea + att_h

    def forward(self, input_x):

        input_x = self.dropout_2d(input_x)
        input_x = input_x.transpose(1, 2).contiguous()

        input_x = input_x.view(-1, 2048, 7, 7)

        mask = self.mask_conv(input_x)
        mask = mask.view(-1, NUM_SEGMENTS, 1, 7, 7)

        input_x = input_x.view(-1, NUM_SEGMENTS, 2048, 7, 7)

        # Calculate total variation regularization (anisotropic version)
        # https://www.wikiwand.com/en/Total_variation_denoising
        diff_i = torch.sum(torch.abs(mask[:, :, :, :, 1:] - mask[:, :, :, :, :-1]))
        diff_j = torch.sum(torch.abs(mask[:, :, :, 1:, :] - mask[:, :, :, :-1, :]))
        tv_loss = TV_REG_FACTOR * (diff_i + diff_j)

        mask_A = (mask > 0.5).type(torch.cuda.FloatTensor)
        mask_B = (mask < 0.5).type(torch.cuda.FloatTensor)
        contrast_loss = -(mask * mask_A).mean(0).sum() * CONTRAST_REG_FACTOR * 0.5 + \
                        (mask * mask_B).mean(0).sum() * CONTRAST_REG_FACTOR * 0.5

        mask_input_x_org = mask * input_x

        h0, c0 = self.get_start_states(mask_input_x_org)

        output_list = []
        temporal_att_weight = None

        for i in range(NUM_SEGMENTS):
            temporal_att_weight_list = []
            for j in range(NUM_SEGMENTS):
                mask_input_x_for_att_per_frame = mask_input_x_org[:, j, :, :, :]
                temporal_att_weight = self.temporal_attention_layer(mask_input_x_for_att_per_frame, h0)

                squeezed_temporal_att_weight = temporal_att_weight.squeeze(dim=1)
                temporal_att_weight_list.append(squeezed_temporal_att_weight.cpu().data.numpy())

            temporal_att_weight = torch.from_numpy(np.asarray(temporal_att_weight_list).squeeze())
            temporal_att_weight = temporal_att_weight.transpose(0, 1).cuda()

            # [30, 50]
            temporal_att_weight = F.softmax(temporal_att_weight, dim=1)
            print("temporal_att_weight: ", temporal_att_weight)
            weighted_mask = mask_input_x_org * (temporal_att_weight.view(-1, NUM_SEGMENTS, 1, 1, 1))
            weighted_mask_input_all_frame = torch.sum(weighted_mask, dim=1)

            h0, c0 = self.conv_lstm_cell(weighted_mask_input_all_frame, (h0, c0))

            output_list.append(torch.mean(torch.mean(h0, dim=3), dim=2))
            temporal_att_weight_list.append(temporal_att_weight)

        output = torch.mean(torch.stack(output_list, dim=0), 0)
        final_output = self.fc_out(output)

        return final_output, temporal_att_weight, mask, tv_loss, contrast_loss


def train(batch_size, train_data, train_label, model, model_optimizer, criterion):
    loss, mask_l1_loss = 0, 0
    model_optimizer.zero_grad()
    logits, att_weight, mask, tv_loss, contrast_loss = model.forward(train_data)
    loss += criterion(logits, train_label)
    att_reg = F.relu(att_weight[:, :-2] * att_weight[:, 2:] - att_weight[:, 1:-1].pow(2)).sqrt().mean()

    # Regularization
    regularization_loss = HP_REG_FACTOR * att_reg
    loss += regularization_loss
    loss += tv_loss
    loss += contrast_loss
    mask_l1_loss += 0.00001 * mask.mean(0).sum()
    loss += mask_l1_loss

    loss.backward()

    model_optimizer.step()

    final_loss = loss.data[0]

    corrects = (torch.max(logits, 1)[1].view(train_label.size()).data == train_label.data).sum()
    train_accuracy = 100.0 * corrects / batch_size
    train_pred_label = torch.max(logits, 1)[1].view(train_label.size()).cpu().data.numpy()

    return mask, train_pred_label, mask_l1_loss, final_loss, regularization_loss, tv_loss, contrast_loss, train_accuracy, att_weight, corrects
