from baseline.layers import *
from hyperparams import get_hyperparams

hyper = get_hyperparams()

batch_size = hyper['batch_size']
num_classes = hyper['num_classes']
img_shape = hyper['image_shape']

class FCDenseNet(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(5, 5, 5, 5, 5),
                 up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48, n_classes=12,
                 dropout=0.5, aleatoric=False):
        super().__init__()
        self.aleatoric = aleatoric

        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []

        # First Convolution
        self.add_module('firstconv', nn.Conv2d(in_channels,
                                               out_chans_first_conv,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1,
                                               bias=True))
        cur_channels_count = out_chans_first_conv

        # Downsampling path
        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i], dropout))

            cur_channels_count += (growth_rate * down_blocks[i])
            skip_connection_channel_counts.insert(0, cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count, dropout))

        # Bottleneck
        self.add_module('bottleneck', Bottlenect(cur_channels_count,
                                                 growth_rate, bottleneck_layers, dropout))
        prev_block_channels = growth_rate * bottleneck_layers
        cur_channels_count += prev_block_channels

        # Upsampling
        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks) - 1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(
                DenseBlock(cur_channels_count, growth_rate, up_blocks[i], dropout, upsample=True))
            prev_block_channels = growth_rate * up_blocks[i]
            cur_channels_count += prev_block_channels

        # Final DenseBlock
        self.transUpBlocks.append(
            TransitionUp(prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(DenseBlock(cur_channels_count, growth_rate, up_blocks[-1], dropout, upsample=False))
        cur_channels_count += growth_rate * up_blocks[-1]

        # Softmax
        self.finalConv = nn.Conv2d(cur_channels_count,
                                   n_classes,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.log_var = nn.Conv2d(cur_channels_count,
                                 1,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)


    def forward(self, x):
        log_var = None
        out = self.firstconv(x)
        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        out_ = self.finalConv(out)
        if self.aleatoric:
            log_var = self.log_var(out)
        out_ = out_.view(batch_size, num_classes, img_shape[0], img_shape[1])
        out_ = self.softmax(out_)
        return out_, log_var

def FCDenseNet57(n_classes, dropout=0.5):
    return FCDenseNet(
        in_channels=3, down_blocks=(4, 4, 4, 4, 4),
        up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=4,
        growth_rate=12, out_chans_first_conv=48, n_classes=n_classes, dropout=dropout)


def FCDenseNet67(n_classes, dropout=0.5):
    return FCDenseNet(
        in_channels=3, down_blocks=(5, 5, 5, 5, 5),
        up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
        growth_rate=16, out_chans_first_conv=48, n_classes=n_classes, dropout=dropout)


def FCDenseNet103(n_classes, dropout=0.5):
    return FCDenseNet(
        in_channels=3, down_blocks=(4,5,7,10,12),
        up_blocks=(12,10,7,5,4), bottleneck_layers=15,
        growth_rate=16, out_chans_first_conv=48, n_classes=n_classes, dropout=dropout)

def FCDenseNet57_aleatoric(n_classes, dropout=0.5):
    return FCDenseNet(
        in_channels=3, down_blocks=(4, 4, 4, 4, 4),
        up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=4,
        growth_rate=12, out_chans_first_conv=48, n_classes=n_classes, dropout=dropout, aleatoric=True)

def FCDenseNet67_aleatoric(n_classes, dropout=0.5):
    return FCDenseNet(
        in_channels=3, down_blocks=(5, 5, 5, 5, 5),
        up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
        growth_rate=16, out_chans_first_conv=48, n_classes=n_classes, dropout=dropout, aleatoric=True)

def FCDenseNet103_aleatoric(n_classes, dropout=0.5):
    """Add Heteroscedastic Aleatoric Uncertainty"""
    return FCDenseNet(
        in_channels=3, down_blocks=(4,5,7,10,12),
        up_blocks=(12,10,7,5,4), bottleneck_layers=15,
        growth_rate=16, out_chans_first_conv=48, n_classes=n_classes, dropout=dropout, aleatoric=True)