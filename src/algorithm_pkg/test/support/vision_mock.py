"""模拟视觉输入：固定延迟、位置噪声、丢帧。

真实相机中继未交付前，用它把“视觉会晚、会抖、会丢”这三件事
注入到纯 Python 测试里。时间戳保留源头时刻，和真实网络延迟一致；重打
时间戳的 relay 不算延迟，测不出门禁行为。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

TruthFn = Callable[[float], Tuple[float, float, float]]


@dataclass
class MockVision:
    """按帧生成视觉采样。

    Attributes:
        delay: 从采样到送达的固定延迟，秒。
        noise_sigma: 每个坐标轴的高斯噪声标准差，米。
        dropout: 丢帧窗口 ``(start, end)``，秒；按源头时刻判断。
        seed: 随机种子；每个实例独立，保证用例可重复。
    """

    delay: float = 0.0
    noise_sigma: float = 0.0
    dropout: Tuple[Tuple[float, float], ...] = ()
    seed: int = 20261123
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.delay < 0.0:
            raise ValueError("delay must be non-negative")
        if self.noise_sigma < 0.0:
            raise ValueError("noise_sigma must be non-negative")
        self._rng = random.Random(self.seed)

    def sample(
        self,
        now: float,
        truth: TruthFn,
    ) -> Optional[Tuple[float, Tuple[float, float, float]]]:
        """返回 ``(源时刻, 位置)``；丢帧或源头时刻无效时返回 ``None``。"""
        source = now - self.delay
        if source < 0.0:
            return None
        for start, end in self.dropout:
            if start <= source < end:
                return None

        x, y, z = truth(source)
        if self.noise_sigma > 0.0:
            x += self._rng.gauss(0.0, self.noise_sigma)
            y += self._rng.gauss(0.0, self.noise_sigma)
            z += self._rng.gauss(0.0, self.noise_sigma)
        return source, (x, y, z)
