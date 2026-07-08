"""
Level 1 状态机——简单的顺序状态流转。

IDLE → DETECT → PLAN → PRE_GRASP → GRASP → POST_GRASP → PRE_PLACE → PLACE → VERIFY
                                                             ↑                        │
                                                             └──(还有盘子)──────────────┘
Level 1 无异常处理、无重试。异常恢复留给 Level 2。
"""

from __future__ import annotations

from enum import Enum, auto


class State(Enum):
    """任务状态枚举。"""
    IDLE = auto()          # 空闲/初始化
    DETECT = auto()        # 检测下一个盘子
    PLAN = auto()          # 生成抓取+放置姿态
    PRE_GRASP = auto()     # 移动到预抓取位置
    GRASP = auto()         # 闭合夹爪
    POST_GRASP = auto()    # 抬升盘子
    PRE_PLACE = auto()     # 移动到卡槽上方
    PLACE = auto()         # 释放盘子到卡槽
    VERIFY = auto()        # 检查是否还有盘子
    DONE = auto()          # 所有盘子已处理
    FAILED = auto()        # 失败（Level 2+ 使用）


class StateMachine:
    """Level 1 简单顺序状态机。

    用法:
        sm = StateMachine()
        while sm.current != State.DONE:
            match sm.current:
                case State.DETECT:
                    ...
                    sm.advance()
    """

    # Level 1 正常流转顺序
    SEQUENCE = [
        State.DETECT,
        State.PLAN,
        State.PRE_GRASP,
        State.GRASP,
        State.POST_GRASP,
        State.PRE_PLACE,
        State.PLACE,
        State.VERIFY,
    ]

    def __init__(self):
        self._current = State.IDLE
        self._seq_idx = 0
        self._total_plates = 0
        self._processed = 0

    @property
    def current(self) -> State:
        return self._current

    @property
    def processed_count(self) -> int:
        return self._processed

    @property
    def is_running(self) -> bool:
        return self._current not in (State.DONE, State.FAILED)

    def start(self, total_plates: int):
        """开始任务。

        Args:
            total_plates: 需要处理的盘子总数
        """
        self._total_plates = total_plates
        self._processed = 0
        self._seq_idx = 0
        self._current = State.DETECT

    def advance(self):
        """推进到下一个状态。"""
        if self._current == State.DETECT and self._processed >= self._total_plates:
            self._current = State.DONE
            return

        if self._current == State.VERIFY:
            self._processed += 1
            if self._processed >= self._total_plates:
                self._current = State.DONE
            else:
                self._current = State.DETECT
            return

        # 正常推进
        self._seq_idx += 1
        if self._seq_idx < len(self.SEQUENCE):
            self._current = self.SEQUENCE[self._seq_idx]
        else:
            self._seq_idx = 0
            self._current = State.DETECT

    def reset(self):
        """重置状态机。"""
        self._current = State.IDLE
        self._seq_idx = 0
        self._processed = 0

    def fail(self):
        """进入失败状态（Level 2+ 使用）。"""
        self._current = State.FAILED
