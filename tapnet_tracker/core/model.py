#!/usr/bin/env python3
"""
TAPNext Model Definition - JAX/Flax Implementation
包含TAPNext模型的所有组件和相关工具函数
"""

import io
import os
import einops
import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np

# ============================================================================
# TAPNext Model Components
# ============================================================================

class MlpBlock(nn.Module):
    @nn.compact
    def __call__(self, x):
        d = x.shape[-1]
        x = nn.gelu(nn.Dense(4 * d)(x))
        return nn.Dense(d)(x)

class ViTBlock(nn.Module):
    num_heads: int = 12
    
    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm()(x)
        y = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(y, y)
        x = x + y
        y = nn.LayerNorm()(x)
        y = MlpBlock()(y)
        x = x + y
        return x

class Einsum(nn.Module):
    width: int = 768
    
    def setup(self):
        self.w = self.param(
            "w", nn.initializers.zeros_init(), (2, self.width, self.width * 4)
        )
        self.b = self.param(
            "b", nn.initializers.zeros_init(), (2, 1, 1, self.width * 4)
        )[:, 0]
    
    def __call__(self, x):
        return jnp.einsum("...d,cdD->c...D", x, self.w) + self.b

class RMSNorm(nn.Module):
    width: int = 768
    
    def setup(self):
        self.scale = self.param("scale", nn.initializers.zeros_init(), (self.width))
    
    def __call__(self, x):
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed_x = x * jax.lax.rsqrt(var + 1e-6)
        scale = jnp.expand_dims(self.scale, axis=range(len(x.shape) - 1))
        return normed_x * (scale + 1)

class Conv1D(nn.Module):
    width: int = 768
    kernel_size: int = 4
    
    def setup(self):
        self.w = self.param(
            "w", nn.initializers.zeros_init(), (self.kernel_size, self.width)
        )
        self.b = self.param("b", nn.initializers.zeros_init(), (self.width))
    
    def __call__(self, x, state):
        if state is None:
            state = jnp.zeros(
                (x.shape[0], self.kernel_size - 1, x.shape[1]), dtype=x.dtype
            )
        x = jnp.concatenate([state, x[:, None]], axis=1)
        out = (x * self.w[None]).sum(axis=-2) + self.b[None]
        state = x[:, 1 - self.kernel_size :]
        return out, state

class BlockDiagonalLinear(nn.Module):
    width: int = 768
    num_heads: int = 12
    
    def setup(self):
        width = self.width // self.num_heads
        self.w = self.param(
            "w", nn.initializers.zeros_init(), (self.num_heads, width, width)
        )
        self.b = self.param(
            "b", nn.initializers.zeros_init(), (self.num_heads, width)
        )
    
    def __call__(self, x):
        x = einops.rearrange(x, "... (h i) -> ... h i", h=self.num_heads)
        y = jnp.einsum("... h i, h i j -> ... h j", x, self.w) + self.b
        return einops.rearrange(y, "... h j -> ... (h j)", h=self.num_heads)

class RGLRU(nn.Module):
    width: int = 768
    num_heads: int = 12
    
    def setup(self):
        self.a_real_param = self.param(
            "a_param", nn.initializers.zeros_init(), (self.width)
        )
        self.input_gate = BlockDiagonalLinear(
            self.width, self.num_heads, name="input_gate"
        )
        self.a_gate = BlockDiagonalLinear(self.width, self.num_heads, name="a_gate")
    
    def __call__(self, x, state):
        gate_x = jnn.sigmoid(self.input_gate(x))
        if state is None:
            return gate_x * x
        else:
            gate_a = jnn.sigmoid(self.a_gate(x))
            log_a = -8.0 * gate_a * jnn.softplus(self.a_real_param)
            a = jnp.exp(log_a)
            scale_factor = jnp.sqrt(1 - jnp.exp(2 * log_a))
            return a * state + gate_x * x * scale_factor

class MLPBlock(nn.Module):
    width: int = 768
    
    def setup(self):
        self.ffw_up = Einsum(self.width, name="ffw_up")
        self.ffw_down = nn.Dense(self.width, name="ffw_down")
    
    def __call__(self, x):
        out = self.ffw_up(x)
        return self.ffw_down(nn.gelu(out[0]) * out[1])

class RecurrentBlock(nn.Module):
    width: int = 768
    num_heads: int = 12
    kernel_size: int = 4
    
    def setup(self) -> None:
        self.linear_y = nn.Dense(self.width, name="linear_y")
        self.linear_x = nn.Dense(self.width, name="linear_x")
        self.conv_1d = Conv1D(self.width, self.kernel_size, name="conv_1d")
        self.lru = RGLRU(self.width, self.num_heads, name="rg_lru")
        self.linear_out = nn.Dense(self.width, name="linear_out")
    
    def __call__(self, x, state):
        y = jax.nn.gelu(self.linear_y(x))
        x = self.linear_x(x)
        x, conv1d_state = self.conv_1d(
            x, None if state is None else state["conv1d_state"]
        )
        rg_lru_state = self.lru(x, None if state is None else state["rg_lru_state"])
        x = self.linear_out(rg_lru_state * y)
        return x, {"rg_lru_state": rg_lru_state, "conv1d_state": conv1d_state}

class ResidualBlock(nn.Module):
    width: int = 768
    num_heads: int = 12
    kernel_size: int = 4
    
    def setup(self):
        self.temporal_pre_norm = RMSNorm(self.width)
        self.recurrent_block = RecurrentBlock(
            self.width, self.num_heads, self.kernel_size, name="recurrent_block"
        )
        self.channel_pre_norm = RMSNorm(self.width)
        self.mlp = MLPBlock(self.width, name="mlp_block")
    
    def __call__(self, x, state):
        y = self.temporal_pre_norm(x)
        y, state = self.recurrent_block(y, state)
        x = x + y
        y = self.mlp(self.channel_pre_norm(x))
        x = x + y
        return x, state

class ViTSSMBlock(nn.Module):
    width: int = 768
    num_heads: int = 12
    kernel_size: int = 4
    
    def setup(self):
        self.ssm_block = ResidualBlock(self.width, self.num_heads, self.kernel_size)
        self.vit_block = ViTBlock(self.num_heads)
    
    def __call__(self, x, state):
        b = x.shape[0]
        x = einops.rearrange(x, "b n c -> (b n) c")
        x, state = self.ssm_block(x, state)
        x = einops.rearrange(x, "(b n) c -> b n c", b=b)
        x = self.vit_block(x)
        return x, state

class ViTSSMBackbone(nn.Module):
    width: int = 768
    num_heads: int = 12
    kernel_size: int = 4
    num_blocks: int = 12
    
    def setup(self):
        self.blocks = [
            ViTSSMBlock(
                self.width,
                self.num_heads,
                self.kernel_size,
                name=f"encoderblock_{i}",
            )
            for i in range(self.num_blocks)
        ]
        self.encoder_norm = nn.LayerNorm()
    
    def __call__(self, x, state):
        new_states = []
        for i in range(self.num_blocks):
            x, new_state = self.blocks[i](x, None if state is None else state[i])
            new_states.append(new_state)
        x = self.encoder_norm(x)
        return x, new_states

def posemb_sincos_2d(h, w, width):
    """Compute 2D sine-cosine positional embeddings following MoCo v3 logic."""
    y, x = jnp.mgrid[0:h, 0:w]
    freqs = jnp.linspace(0, 1, num=width // 4, endpoint=True)
    inv_freq = 1.0 / (10_000**freqs)
    y = jnp.einsum("h w, d -> h w d", y, inv_freq)
    x = jnp.einsum("h w, d -> h w d", x, inv_freq)
    pos_emb = jnp.concatenate(
        [jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=-1
    )
    return pos_emb

class Backbone(nn.Module):
    width: int = 768
    num_heads: int = 12
    kernel_size: int = 4
    num_blocks: int = 12
    
    def setup(self):
        self.lin_proj = nn.Conv(
            self.width,
            (1, 8, 8),
            strides=(1, 8, 8),
            padding="VALID",
            name="embedding",
        )
        self.mask_token = self.param(
            "mask_token", nn.initializers.zeros_init(), (1, 1, 1, self.width)
        )[:, 0]
        self.unknown_token = self.param(
            "unknown_token", nn.initializers.zeros_init(), (1, 1, self.width)
        )
        self.point_query_token = self.param(
            "point_query_token", nn.initializers.zeros_init(), (1, 1, 1, self.width)
        )[:, 0]
        self.image_pos_emb = self.param(
            "pos_embedding",
            nn.initializers.zeros_init(),
            (1, 256 // 8 * 256 // 8, self.width),
        )
        self.encoder = ViTSSMBackbone(
            self.width,
            self.num_heads,
            self.kernel_size,
            self.num_blocks,
            name="Transformer",
        )
    
    def __call__(self, frame, query_points, step, state):
        x = self.lin_proj(frame)
        b, h, w, c = x.shape
        query_points = jnp.concatenate(
            [query_points[..., :1] - step, query_points[..., 1:]], axis=-1
        )
        posemb2d = posemb_sincos_2d(256, 256, self.width)
        
        def interp(x, y):
            return jax.scipy.ndimage.map_coordinates(
                x, y.T - 0.5, order=1, mode="nearest"
            )
        
        interp_fn = jax.vmap(interp, in_axes=(-1, None), out_axes=-1)
        interp_fn = jax.vmap(interp_fn, in_axes=(None, 0), out_axes=0)
        point_tokens = self.point_query_token + interp_fn(
            posemb2d, query_points[..., 1:]
        )
        
        query_timesteps = query_points[..., 0:1].astype(jnp.int32)
        query_tokens = jnp.where(
            query_timesteps > 0, self.unknown_token, self.mask_token
        )
        query_tokens = jnp.where(
            query_timesteps == 0, point_tokens, query_tokens
        )
        
        image_tokens = (
            jnp.reshape(x, [b, h * w, c]) + self.image_pos_emb
        )
        
        x = jnp.concatenate(
            [image_tokens, query_tokens], axis=-2
        )
        x, state = self.encoder(x, state)
        _, q, _ = query_points.shape
        x = x[:, -q:, :]
        
        return x, state

class TAPNext(nn.Module):
    width: int = 768
    num_heads: int = 12
    kernel_size: int = 4
    num_blocks: int = 12
    
    def setup(self):
        self.backbone = Backbone(
            self.width, self.num_heads, self.kernel_size, self.num_blocks
        )
        self.visible_head = nn.Sequential([
            nn.Dense(256),
            nn.LayerNorm(),
            nn.gelu,
            nn.Dense(256),
            nn.LayerNorm(),
            nn.gelu,
            nn.Dense(1),
        ])
        self.coordinate_head = nn.Sequential([
            nn.Dense(256),
            nn.LayerNorm(),
            nn.gelu,
            nn.Dense(256),
            nn.LayerNorm(),
            nn.gelu,
            nn.Dense(512),
        ])
    
    @nn.compact
    def __call__(self, frame, query_points, step, state):
        feat, state = self.backbone(frame, query_points, step, state)
        track_logits = self.coordinate_head(feat)
        visible_logits = self.visible_head(feat)
        
        position_x, position_y = jnp.split(track_logits, 2, axis=-1)
        position = jnp.stack([position_x, position_y], axis=-2)
        index = jnp.arange(position.shape[-1])[None, None, None]
        argmax = jnp.argmax(position, axis=-1, keepdims=True)
        mask = jnp.abs(argmax - index) <= 20
        probs = jnn.softmax(position * 0.5, axis=-1) * mask
        probs = probs / jnp.sum(probs, axis=-1, keepdims=True)
        tracks = jnp.sum(probs * index, axis=-1) + 0.5
        visible = (visible_logits > 0).astype(jnp.float32)
        return tracks, visible, state

# ============================================================================
# Model Utilities
# ============================================================================

def npload(fname):
    """加载numpy数据"""
    if os.path.exists(fname):
        loaded = np.load(fname, allow_pickle=False)
    else:
        with open(fname, "rb") as f:
            data = f.read()
        loaded = np.load(io.BytesIO(data), allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        return loaded
    else:
        return dict(loaded)

def recover_tree(flat_dict):
    """从扁平字典恢复树结构"""
    tree = {}
    for k, v in flat_dict.items():
        parts = k.split("/")
        node = tree
        for part in parts[:-1]:
            if part not in node:
                node[part] = {}
            node = node[part]
        node[parts[-1]] = v
    return tree

# ============================================================================
# Model Instance and Forward Function
# ============================================================================

# 创建模型实例
model = TAPNext()

@jax.jit
def forward(params, frame, query_points, step, state):
    """TAPNext模型前向传播的JIT编译函数"""
    tracks, visible, state = model.apply(
        {"params": params}, frame, query_points, step, state
    )
    return tracks, visible, state

# ============================================================================
# Model Factory Functions
# ============================================================================

def create_tapnext_model(width=768, num_heads=12, kernel_size=4, num_blocks=12):
    """创建自定义配置的TAPNext模型"""
    return TAPNext(
        width=width,
        num_heads=num_heads, 
        kernel_size=kernel_size,
        num_blocks=num_blocks
    )

def get_model_info():
    """获取模型信息"""
    return {
        "name": "TAPNext",
        "framework": "JAX/Flax",
        "default_width": 768,
        "default_num_heads": 12,
        "default_kernel_size": 4,
        "default_num_blocks": 12,
        "input_size": (256, 256),
        "output_format": "[points, frames, batch, 3] - (x, y, visibility)"
    } 