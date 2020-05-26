""" Adapted from the original implementation. """

import collections
from typing import List

import torch

VoVNet19_slim_dw = {
    "stem_out": 64,
    "stage_conv_ch": [64, 80, 96, 112],
    "stage_out_ch": [112, 256, 384, 512],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "dw": True,
}

VoVNet19_dw = {
    "stem_out": 64,
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "dw": True,
}

VoVNet19_slim = {
    "stem_out": 128,
    "stage_conv_ch": [64, 80, 96, 112],
    "stage_out_ch": [112, 256, 384, 512],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "dw": False,
}

VoVNet19 = {
    "stem_out": 128,
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "dw": False,
}

VoVNet39 = {
    "stem_out": 128,
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 5,
    "block_per_stage": [1, 1, 2, 2],
    "dw": False,
}

VoVNet57 = {
    "stem_out": 128,
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 5,
    "dw": False,
}

VoVNet99 = {
    "stem_out": 128,
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 5,
    "block_per_stage": [1, 3, 9, 3],
    "dw": False,
}

_STAGE_SPECS = {
    "vovnet-19-slim-dw": VoVNet19_slim_dw,
    "vovnet-19-dw": VoVNet19_dw,
    "vovnet-19-slim": VoVNet19_slim,
    "vovnet-19": VoVNet19,
    "vovnet-39": VoVNet39,
    "vovnet-57": VoVNet57,
    "vovnet-99": VoVNet99,
}


def dw_conv(in_channels: int, out_channels: int) -> List[torch.nn.Module]:
    """ Depthwise separable pointwise linear convolution. """
    return [
        torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        ),
        torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=True,
        ),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=True),
    ]


def conv(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    groups: int = 1,
    kernel_size: int = 3,
    padding: int = 1,
) -> List[torch.nn.Module]:
    """ 3x3 convolution with padding """
    return [
        torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=True),
    ]


def pointwise(in_channels: int, out_channels: int) -> List[torch.nn.Module]:
    """ Pointwise convolution. """
    return [
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=True),
    ]


class eSE(torch.nn.Module):
    """ This is adapted from the efficientnet Squeeze Excitation. The idea is not 
    squeezing the number of channels keeps more information. """

    def __init__(self, channel: int) -> None:
        super().__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Conv2d(
            channel, channel, kernel_size=1, padding=0
        )  # (Linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.avg_pool(x)
        out = self.fc(out)
        out = torch.nn.functional.relu6(out + 3.0, inplace=True) / 6.0
        return out * x


class OSA(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        stage_channels: int,
        concat_channels: int,
        layer_per_block: int,
        identity: bool = False,
        conv_op=conv,
    ) -> None:

        super().__init__()
        original_channels = in_channels
        self.identity = identity
        self.isReduced = False
        self.layers = torch.nn.ModuleList()
        self.depthwise = type(conv_op) == dw_conv

        if self.depthwise and in_channels != stage_channels:
            self.isReduced = True
            self.conv_reduction = torch.nn.Sequential(
                *pointwise(in_channels, stage_channels)
            )
        for _ in range(layer_per_block):
            self.layers.append(
                torch.nn.Sequential(*conv_op(in_channels, stage_channels))
            )
            in_channels = stage_channels

        # feature aggregation
        aggregated = original_channels + layer_per_block * stage_channels
        self.concat = torch.nn.Sequential(*pointwise(aggregated, concat_channels))
        self.ese = eSE(concat_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.identity:
            identity_feat = x

        output = [x]
        if self.depthwise and self.isReduced:
            x = self.conv_reduction(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        xt = self.concat(x)

        xt = self.ese(xt)

        if self.identity:
            xt = xt + identity_feat

        return xt


class _OSA_stage(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        stage_ch: int,
        concat_ch: int,
        block_per_stage: int,
        layer_per_block: int,
        stage_num: int,
        depthwise: bool = False,
    ) -> None:
        super().__init__()

        if not stage_num == 2:
            self.add_module(
                "Pooling", torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
            )

        for idx in range(block_per_stage):
            self.add_module(
                f"OSA{stage_num}_{idx + 1}",
                OSA(
                    in_channels if idx == 0 else concat_ch,
                    stage_ch,
                    concat_ch,
                    layer_per_block,
                    identity=False if idx == 0 else True,
                    conv_op=depthwise,
                ),
            )


class VoVNet(torch.nn.Sequential):
    def __init__(
        self, model_name: str, num_classes: int = 10, input_channels: int = 3
    ) -> None:
        """
        Args:
            model_name: Which model to create.
            num_classes: The number of classification classes.
            input_channels: The number of input channels.
        Usage:
        >>> net = VoVNet("V-19-slim-dw", num_classes=1000)
        >>> with torch.no_grad():
        ...    out = net(torch.randn(1, 3, 512, 512))
        >>> print(out.shape)
        torch.Size([1, 1000])
        """
        super().__init__()
        assert model_name in _STAGE_SPECS, f"{model_name} not supported."

        self.model_name = model_name
        stem_ch = _STAGE_SPECS[model_name]["stem_out"]
        config_stage_ch = _STAGE_SPECS[model_name]["stage_conv_ch"]
        config_concat_ch = _STAGE_SPECS[model_name]["stage_out_ch"]
        block_per_stage = _STAGE_SPECS[model_name]["block_per_stage"]
        layer_per_block = _STAGE_SPECS[model_name]["layer_per_block"]
        conv_type = dw_conv if _STAGE_SPECS[model_name]["dw"] else conv

        # Construct the stem.
        stem = conv(input_channels, 64)
        stem += conv(64, 64)
        stem += conv(64, stem_ch)
        self.model = torch.nn.Sequential()
        self.model.add_module("stem", torch.nn.Sequential(*stem))
        self._out_feature_channels = [stem_ch]

        stem_out_ch = [stem_ch]
        in_ch_list = stem_out_ch + config_concat_ch[:-1]

        # Add the OSA modules. Typically 4 modules.
        for idx in range(len(config_stage_ch)):
            self.model.add_module(
                f"OSA_{(idx + 2)}",
                _OSA_stage(
                    in_ch_list[idx],
                    config_stage_ch[idx],
                    config_concat_ch[idx],
                    block_per_stage[idx],
                    layer_per_block,
                    idx + 2,
                    conv_type,
                ),
            )

            self._out_feature_channels.append(config_concat_ch[idx])

        # Add the classification head.
        self.model.add_module(
            "classifier",
            torch.nn.Sequential(
                torch.nn.BatchNorm2d(self._out_feature_channels[-1]),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(self._out_feature_channels[-1], num_classes, bias=True),
            ),
        )

        # Initialize weights.
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def forward_pyramids(self, x: torch.Tensor) -> collections.OrderedDict:
        """
        Args:
            model_name: Which model to create.
            num_classes: The number of classification classes.
            input_channels: The number of input channels.
        Usage:
        >>> net = VoVNet("V-19-slim-dw", num_classes=1000)
        >>> net.delete_classification_head()
        >>> with torch.no_grad():
        ...    out = net.forward_pyramids(torch.randn(1, 3, 512, 512))
        >>> print(out.shape)
        torch.Size([1, 1000])
        """
        levels = collections.OrderedDict()
        levels[1] = self.model.stem(x)
        levels[2] = self.model.OSA_2(levels[1])
        levels[3] = self.model.OSA_3(levels[2])
        levels[4] = self.model.OSA_4(levels[3])
        levels[5] = self.model.OSA_5(levels[4])
        return levels

    def delete_classification_head(self) -> None:
        """ Call this before using model as an object detection backbone. """
        del self.model.classifier