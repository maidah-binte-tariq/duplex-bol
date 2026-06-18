#!/usr/bin/env python3
"""Generate the figures in docs/assets/ from the actual code — reproducibly.

Every figure here is derived from real package behavior, not a mockup:

* ``barge_in_timeline`` runs the real :class:`DuplexOrchestrator` and plots the
  event stream it emits (the 60 ms barge-in stop is measured, not drawn).
* ``stereo_synthesis`` plots the actual stereo waveform that
  :func:`duplex_bol.data.build_dialogue` produces.
* ``architecture`` / ``state_machine`` are laid out by hand but kept in code so
  they version with the design.

    make figures            # or: python scripts/make_figures.py

Anyone can regenerate them; that's the point of committing the generator rather
than just the PNGs.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from duplex_bol.cascade import AudioFrame, BargeIn, DuplexOrchestrator, SpeechStarted
from duplex_bol.cascade.fakes import ChunkedTTS, RuleBasedAgent, ScriptedASR
from duplex_bol.cascade.vad import EnergyVAD
from duplex_bol.data import DialogueConfig, SpeakerClip, build_dialogue

ASSETS = Path(__file__).resolve().parents[1] / "docs" / "assets"

# --- a small, restrained house style -----------------------------------------
INK = "#1b2a3a"  # near-navy text/lines
MUTED = "#6b7a89"  # secondary text
CALLER = "#e8743b"  # warm — the human
AGENT = "#2e6f9e"  # cool — the machine
GREEN = "#4c9a6a"
PANEL = "#eef2f5"
BORDER = "#c3cdd6"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": INK,
        "axes.edgecolor": BORDER,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)


def _save(fig: plt.Figure, name: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(ASSETS / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote docs/assets/{name}.png (+svg)")


def _box(ax, cx, cy, w, h, text, *, fc=PANEL, ec=AGENT, tc=INK, fs=11, bold=False):
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.6,
            edgecolor=ec,
            facecolor=fc,
            zorder=2,
        )
    )
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight="bold" if bold else "normal",
        color=tc,
        zorder=3,
    )


def _arrow(ax, p0, p1, *, color=INK, style="-|>", lw=1.8, ls="-", rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle=style,
            mutation_scale=14,
            lw=lw,
            color=color,
            linestyle=ls,
            shrinkA=2,
            shrinkB=2,
            zorder=1,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


# --- figure 1: system architecture -------------------------------------------
def architecture() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    y = 1.7
    xs = [1.4, 3.7, 6.0, 8.3, 10.6]
    labels = [
        ("caller\naudio", PANEL, MUTED, INK),
        ("VAD\nvoice activity", "#fdeee6", CALLER, INK),
        ("ASR · ear\nspeech → text", PANEL, AGENT, INK),
        ("LLM · brain\nUrdu reply", PANEL, AGENT, INK),
        ("TTS · mouth\ntext → speech", PANEL, AGENT, INK),
    ]
    for x, (lab, fc, ec, tc) in zip(xs, labels):
        _box(ax, x, y, 1.9, 1.05, lab, fc=fc, ec=ec, tc=tc, fs=10.5)
    for x0, x1 in pairwise(xs):
        _arrow(ax, (x0 + 0.95, y), (x1 - 0.95, y))
    # output
    _box(ax, 10.6, y, 1.9, 1.05, "TTS · mouth\ntext → speech", fc=PANEL, ec=AGENT, fs=10.5)
    _arrow(ax, (xs[-1] + 0.95, y), (11.9, y))
    ax.text(11.95, y, "bot\naudio", ha="left", va="center", fontsize=10, color=MUTED)

    # orchestrator on top, spanning, with barge-in cancel
    _box(
        ax,
        6.0,
        3.7,
        7.6,
        1.0,
        "DuplexOrchestrator  ·  synchronous barge-in state machine",
        fc="#eaf1f7",
        ec=AGENT,
        fs=11.5,
        bold=True,
    )
    # VAD feeds the orchestrator; orchestrator cancels TTS
    _arrow(ax, (3.7, y + 0.55), (4.6, 3.2), color=CALLER, ls=(0, (4, 3)), rad=-0.25)
    ax.text(3.0, 2.95, "speech\ndetected", fontsize=8.5, color=CALLER, ha="center")
    _arrow(ax, (7.6, 3.2), (8.3, y + 0.55), color="#c0392b", ls=(0, (4, 3)), rad=-0.25)
    ax.text(9.15, 2.95, "cancel\n(barge-in)", fontsize=8.5, color="#c0392b", ha="center")

    ax.text(0.2, 4.7, "Track B — the cascade", fontsize=13, fontweight="bold", color=INK)
    ax.text(
        0.2,
        0.35,
        "Three swappable model components behind a referee that keeps "
        "listening while the bot talks.",
        fontsize=9.5,
        color=MUTED,
    )
    _save(fig, "architecture")


# --- figure 2: barge-in timeline (REAL orchestrator output) ------------------
def barge_in_timeline() -> None:
    frame_ms = 20.0
    # A scenario with room for the bot to actually speak before being cut off.
    pattern = "SSSS" + "." * 11 + "SSSSS" + "." * 8
    frames = [
        AudioFrame(np.full(320, 0.3, np.float32) if c == "S" else np.zeros(320, np.float32))
        for c in pattern
    ]
    orch = DuplexOrchestrator(
        vad=EnergyVAD(),
        asr=ScriptedASR(["السلام علیکم", "نہیں رکو"]),
        agent=RuleBasedAgent(default="وعلیکم السلام جی فرمائیے میں سن رہا ہوں"),
        tts=ChunkedTTS(frames_per_word=3),
        frame_duration_ms=frame_ms,
    )
    events = list(orch.run(frames))
    speech_started = next(e for e in events if isinstance(e, SpeechStarted))
    barge = next(e for e in events if isinstance(e, BargeIn))

    # caller speech runs straight from the input pattern
    caller_runs, i = [], 0
    while i < len(pattern):
        if pattern[i] == "S":
            j = i
            while j < len(pattern) and pattern[j] == "S":
                j += 1
            caller_runs.append((i * frame_ms, j * frame_ms))
            i = j
        else:
            i += 1
    bot_run = (speech_started.frame_index * frame_ms, barge.frame_index * frame_ms)

    fig, ax = plt.subplots(figsize=(11, 3.7))
    # faint lane guides
    for yc in (0.8, 1.8):
        ax.axhline(yc, color=BORDER, lw=0.8, zorder=0)
    for x0, x1 in caller_runs:
        ax.add_patch(
            Rectangle((x0, 1.56), x1 - x0, 0.48, fc=CALLER, ec="none", alpha=0.92, zorder=3)
        )
    ax.add_patch(
        Rectangle(
            (bot_run[0], 0.56),
            bot_run[1] - bot_run[0],
            0.48,
            fc=AGENT,
            ec="none",
            alpha=0.92,
            zorder=3,
        )
    )

    bx = barge.frame_index * frame_ms
    ax.axvline(bx, color="#c0392b", lw=1.6, ls=(0, (3, 2)))
    # the measured stop latency, as a bracket on the agent lane
    onset = bx - barge.stop_latency_ms
    ax.annotate(
        "",
        xy=(bx, 0.42),
        xytext=(onset, 0.42),
        arrowprops={"arrowstyle": "<|-|>", "color": "#c0392b", "lw": 1.4},
    )
    ax.text(
        (onset + bx) / 2,
        0.24,
        f"barge-in stop\n{barge.stop_latency_ms:.0f} ms",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#c0392b",
        fontweight="bold",
    )

    # response latency bracket on the agent lane
    last_caller = caller_runs[0][1]
    sx = speech_started.frame_index * frame_ms
    ax.annotate(
        "",
        xy=(sx, 1.18),
        xytext=(last_caller, 1.18),
        arrowprops={"arrowstyle": "<|-|>", "color": GREEN, "lw": 1.4},
    )
    ax.text(
        (last_caller + sx) / 2,
        1.30,
        f"response\n{sx - last_caller:.0f} ms",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=GREEN,
        fontweight="bold",
    )

    ax.text(caller_runs[1][0] + 8, 2.18, "caller cuts in", fontsize=9.5, color=CALLER)
    ax.set_yticks([0.82, 1.82])
    ax.set_yticklabels(["Agent", "Caller"], fontsize=11, fontweight="bold")
    ax.set_ylim(0.0, 2.5)
    ax.set_xlim(-10, len(pattern) * frame_ms + 10)
    ax.set_xlabel("time (ms)")
    ax.set_title(
        "Full-duplex barge-in — the agent goes quiet "
        f"{barge.stop_latency_ms:.0f} ms after the caller interrupts",
        fontsize=12.5,
        fontweight="bold",
        loc="left",
        pad=12,
    )
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)
    ax.legend(
        handles=[
            Line2D([0], [0], color=CALLER, lw=8, label="caller speaking"),
            Line2D([0], [0], color=AGENT, lw=8, label="agent speaking"),
        ],
        loc="upper right",
        frameon=False,
        fontsize=9.5,
    )
    _save(fig, "barge_in_timeline")


# --- figure 3: two-party stereo synthesis (REAL waveform) --------------------
def stereo_synthesis() -> None:
    sr = 24_000

    def tone(freq, sec=0.5):
        t = np.arange(int(sec * sr)) / sr
        env = np.minimum(1.0, np.minimum(t * 12, (sec - t) * 12))
        return (0.7 * env * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    clips = [
        SpeakerClip("agent", tone(180, 0.55), "السلام علیکم", sr),
        SpeakerClip("user", tone(250, 0.5), "وعلیکم", sr),
    ]
    stereo, _dialogue = build_dialogue(clips, DialogueConfig(inter_turn_gap_s=0.0, overlap_s=0.18))
    t = np.arange(stereo.shape[0]) / sr

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 4.2), sharex=True)
    a1.plot(t, stereo[:, 0], color=AGENT, lw=0.9)
    a2.plot(t, stereo[:, 1], color=CALLER, lw=0.9)
    overlap = (np.abs(stereo[:, 0]) > 1e-3) & (np.abs(stereo[:, 1]) > 1e-3)
    if overlap.any():
        x0, x1 = t[overlap][0], t[overlap][-1]
        for ax in (a1, a2):
            ax.axvspan(x0, x1, color=GREEN, alpha=0.16, lw=0)
        a1.text(
            (x0 + x1) / 2,
            0.92,
            "overlap\n(simultaneous speech)",
            ha="center",
            va="top",
            fontsize=9,
            color=GREEN,
            fontweight="bold",
        )

    for ax, lab, col in ((a1, "Left  ·  agent", AGENT), (a2, "Right  ·  user", CALLER)):
        ax.set_ylabel(lab, color=col, fontweight="bold", fontsize=10.5)
        ax.set_ylim(-1.05, 1.05)
        ax.set_yticks([])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    a2.set_xlabel("time (s)")
    a1.set_title(
        "Synthesized two-party stereo — one speaker per channel, with deliberate overlap",
        fontsize=12.5,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    _save(fig, "stereo_synthesis")


# --- figure 4: orchestrator state machine ------------------------------------
def state_machine() -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    _box(ax, 2.6, 2.0, 2.7, 1.2, "LISTENING\nASR + VAD", fc="#fdeee6", ec=CALLER, fs=12, bold=True)
    _box(ax, 7.4, 2.0, 2.7, 1.2, "SPEAKING\nstream TTS", fc="#eaf1f7", ec=AGENT, fs=12, bold=True)
    _arrow(ax, (4.0, 2.45), (6.0, 2.45), color=GREEN, rad=-0.32)
    ax.text(
        5.0, 3.32, "caller stops\n(VAD offset, debounced)", ha="center", fontsize=9, color=GREEN
    )
    _arrow(ax, (6.0, 1.55), (4.0, 1.55), color="#c0392b", rad=-0.32)
    ax.text(
        5.0, 0.72, "caller barges in  /  bot finishes", ha="center", fontsize=9, color="#c0392b"
    )
    ax.text(
        0.1,
        3.7,
        "DuplexOrchestrator — two states, debounced edges",
        fontsize=12.5,
        fontweight="bold",
        color=INK,
    )
    _save(fig, "state_machine")


def main() -> None:
    print("generating figures from live code:")
    architecture()
    barge_in_timeline()
    stereo_synthesis()
    state_machine()
    print("done.")


if __name__ == "__main__":
    main()
