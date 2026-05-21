"""
动态语境窗口截取
- 普通文本：实体前后 N 个字符
- 对话场景（IMCS）：实体所在轮次 ± K 轮，保留角色

输出格式：{"entity": str, "context": str, "source": "text"|"dialogue", "loc": ...}
"""
import re
import sys
import os
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import CONTEXT_WINDOW_CHARS, CONTEXT_DIALOGUE_TURNS


def context_from_text(text: str, entity: str,
                      window: int = CONTEXT_WINDOW_CHARS) -> List[Dict]:
    """返回实体在文本中所有出现位置的语境片段。"""
    if not entity or entity not in text:
        return []
    out = []
    for m in re.finditer(re.escape(entity), text):
        s, e = m.start(), m.end()
        ctx = text[max(0, s - window): min(len(text), e + window)]
        out.append({
            "entity": entity, "context": ctx,
            "source": "text", "loc": {"start": s, "end": e},
        })
    return out


def context_from_dialogue(dialogue: List[Dict], entity: str,
                          self_report: str = "",
                          turns: int = CONTEXT_DIALOGUE_TURNS) -> List[Dict]:
    """
    IMCS：定位实体所在轮次，取前后 turns 轮。
    每轮拼接 speaker:sentence；首轮带上 self_report 作为先验。
    若实体出现在 self_report 中，单独生成一条语境。
    """
    out = []
    if self_report and entity in self_report:
        out.append({
            "entity": entity,
            "context": f"主诉: {self_report}",
            "source": "dialogue", "loc": {"turn_index": -1},
        })

    for ti, turn in enumerate(dialogue):
        sent = turn.get("sentence", "")
        if entity not in sent:
            continue
        lo = max(0, ti - turns)
        hi = min(len(dialogue), ti + turns + 1)
        lines = []
        if lo == 0 and self_report:
            lines.append(f"[主诉] {self_report}")
        for tj in range(lo, hi):
            t = dialogue[tj]
            tag = "*" if tj == ti else " "
            lines.append(f"{tag}{t.get('speaker','?')}: {t.get('sentence','')}")
        out.append({
            "entity": entity,
            "context": "\n".join(lines),
            "source": "dialogue",
            "loc": {"turn_index": ti, "speaker": turn.get("speaker", "")},
        })
    return out
