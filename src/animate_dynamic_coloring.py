#!/usr/bin/env python3
"""Animate dynamic graph coloring and save as GIF or MP4.

Usage example:
  python src/animate_dynamic_coloring.py --out results/dynamic_coloring_demo.gif --nodes 30 --frames 40

The script will attempt to use Matplotlib's PillowWriter for GIFs or FFMpegWriter for MP4.
If those are not available, it will fall back to imageio if installed.
"""
import argparse
import os
import random
from typing import List, Dict

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def generate_dynamic_graph(num_nodes: int, num_edges: int, steps: int, seed: int = None) -> List[nx.Graph]:
    rng = random.Random(seed)
    G = nx.gnm_random_graph(num_nodes, num_edges, seed=seed)
    snapshots = [G.copy()]
    for _ in range(steps - 1):
        G = snapshots[-1].copy()
        if rng.random() < 0.5 and G.number_of_edges() > 0:
            e = rng.choice(list(G.edges()))
            G.remove_edge(*e)
        else:
            n1, n2 = rng.sample(range(num_nodes), 2)
            G.add_edge(n1, n2)
        snapshots.append(G)
    return snapshots


def compute_colorings(snapshots: List[nx.Graph]) -> List[Dict[int, int]]:
    # For each snapshot compute a greedy coloring (may change across snapshots)
    colorings = []
    for G in snapshots:
        coloring = nx.coloring.greedy_color(G, strategy="largest_first")
        # Ensure every node has a color (isolated nodes)
        for n in G.nodes():
            if n not in coloring:
                coloring[n] = 0
        colorings.append(coloring)
    return colorings


def draw_frame(ax, G, pos, coloring, cmap=plt.cm.tab10):
    ax.clear()
    ax.set_axis_off()
    nodes = list(G.nodes())
    colors = [coloring.get(n, 0) for n in nodes]
    nx.draw_networkx_edges(G, pos=pos, ax=ax, alpha=0.6)
    nx.draw_networkx_nodes(G, pos=pos, nodelist=nodes, node_color=colors,
                           cmap=cmap, node_size=250, ax=ax)
    nx.draw_networkx_labels(G, pos=pos, labels={n: n for n in nodes}, font_size=7, ax=ax)


def save_animation_by_writer(fig, anim, out_path, fps=10):
    # Choose writer by extension
    ext = os.path.splitext(out_path)[1].lower()
    try:
        if ext in ('.gif',):
            from matplotlib.animation import PillowWriter

            writer = PillowWriter(fps=fps)
            anim.save(out_path, writer=writer)
        else:
            from matplotlib.animation import FFMpegWriter

            writer = FFMpegWriter(fps=fps)
            anim.save(out_path, writer=writer)
        print(f"Saved animation to {out_path}")
        return True
    except Exception as e:
        print(f"Writer save failed: {e}")
        return False


def save_animation_via_imageio(frames: List[np.ndarray], out_path: str, fps: int = 10):
    try:
        import imageio

        imageio.mimsave(out_path, frames, fps=fps)
        print(f"Saved animation to {out_path} via imageio")
        return True
    except Exception as e:
        print(f"imageio save failed: {e}")
        return False


def run_demo(out_path: str, nodes: int, edges: int, frames: int, fps: int, seed: int):
    snapshots = generate_dynamic_graph(nodes, edges, frames, seed=seed)
    colorings = compute_colorings(snapshots)

    # Use the layout of the last snapshot (keeps positions stable)
    pos = nx.spring_layout(snapshots[-1], seed=seed)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Try to use FuncAnimation + writer first; otherwise collect frames and fallback to imageio
    try:
        from matplotlib.animation import FuncAnimation

        def update(i):
            draw_frame(ax, snapshots[i], pos, colorings[i])

        anim = FuncAnimation(fig, update, frames=len(snapshots), interval=1000 // max(1, fps))
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        ok = save_animation_by_writer(fig, anim, out_path, fps=fps)
        if not ok:
            raise RuntimeError('Writer failed')
        plt.close(fig)
        return True
    except Exception as e:
        # Fallback: render frames to arrays then use imageio
        print(f"Falling back to frame buffer method: {e}")
        frames_buf = []
        for i in range(len(snapshots)):
            draw_frame(ax, snapshots[i], pos, colorings[i])
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            data = data.reshape((h, w, 3))
            frames_buf.append(data)
        plt.close(fig)
        return save_animation_via_imageio(frames_buf, out_path, fps=fps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='results/dynamic_coloring_demo.gif', help='Output file (gif or mp4)')
    p.add_argument('--nodes', type=int, default=30)
    p.add_argument('--edges', type=int, default=90)
    p.add_argument('--frames', type=int, default=30)
    p.add_argument('--fps', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    ok = run_demo(args.out, args.nodes, args.edges, args.frames, args.fps, args.seed)
    if not ok:
        print('\nFailed to save animation automatically.\nSuggestions:')
        print('- Ensure `pillow` is installed for GIF output: pip install pillow')
        print('- For MP4 output ensure `ffmpeg` is installed and Matplotlib detects it')
        print('- Or install `imageio` (pip install imageio) for fallback saving')


if __name__ == '__main__':
    main()
