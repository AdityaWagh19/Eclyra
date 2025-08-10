"""
Eclyra - Emotional Playlist Storyteller

A sophisticated AI-powered tool that transforms your Spotify playlists into emotional journeys with custom cover art and intelligent song ordering.

Author: Aditya Wagh
License: MIT
Repository: https://github.com/yourusername/eclyra
"""

# Eclyra - Emotional Playlist Storyteller

import atexit
import base64
import io
import json
import os
import random
import re
import sys
import time
import urllib.parse as up
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import google.generativeai as genai
import lyricsgenius
import numpy as np
import requests
import spotipy
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyOAuth
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


# --- Configuration ---
class Config:
    """Configuration class for API keys and settings."""
    
    # API Credentials - Replace with your own
    SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "your_spotify_client_id_here")
    SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "your_spotify_client_secret_here")
    GENIUS_ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN", "your_genius_token_here")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_gemini_api_key_here")
    SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8080/callback")
    
    # Application Settings
    MAX_IMAGE_SIZE_BYTES = 256 * 1024
    DEFAULT_IMAGE_SIZE = (1080, 1080)
    MAX_LYRICS_LENGTH = 350
    DEFAULT_MARKET = "US"


# --- Global ---
console = Console()


def retry_net():
    """Network retry decorator for API calls."""
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
        retry=retry_if_exception_type((
            requests.exceptions.RequestException,
            json.JSONDecodeError,
            SpotifyException,
            RuntimeError,
        ))
    )


# --- Harmonious Cover Art Palettes ---
PREDEFINED_PALETTES = [
    ["#8BC6EC", "#9599E2", "#A1FFCE"], ["#FF9A8B", "#FF6A88", "#FF99AC"],
    ["#FAD0C4", "#FFD1FF", "#B8C0FF"], ["#A9C9FF", "#FFBBEC", "#BDE0FE"],
    ["#764BA2", "#667EEA", "#2AF598"], ["#F6D365", "#FDA085", "#FF758C"],
    ["#20E2D7", "#F9C851", "#FF8A00"], ["#00DBDE", "#FC00FF", "#3F51B5"]
]


# --- Data Models ---
@dataclass
class Song:
    """Represents a song with metadata and lyrics."""
    id: str
    name: str
    artists: List[str]
    lyrics: Optional[str] = None
    popularity: Optional[int] = None
    album_name: Optional[str] = None
    release_year: Optional[str] = None

    def to_dict(self, include_lyrics: bool = True) -> Dict:
        """Convert song to dictionary format."""
        d = {
            "id": self.id,
            "name": self.name,
            "artists": self.artists,
            "popularity": self.popularity,
            "album_name": self.album_name,
            "release_year": self.release_year,
        }
        if include_lyrics:
            d["lyrics"] = self.lyrics
        return {k: v for k, v in d.items() if v is not None}


class StageModel(BaseModel):
    """Pydantic model for playlist stages."""
    name: str
    description: str
    sorted_song_ids: List[str] = Field(default_factory=list)


class PlaylistPlan(BaseModel):
    """Pydantic model for complete playlist plan."""
    story_arc: List[StageModel]
    playlist_names: List[str]
    playlist_description: str


# --- Cover Art Generator ---
class CreativeCoverArtGenerator:
    """Generates beautiful cover art for playlists."""
    
    def __init__(self):
        self.font_path = self._find_font()

    def _find_font(self) -> Optional[str]:
        """Find the best available font on the system."""
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "/Library/Fonts/Arial.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
        for p in candidates:
            if Path(p).exists():
                console.print(f"[dim]Using font: {Path(p).name}[/dim]", highlight=False)
                return p
        console.print("[yellow]Warning: Preferred fonts not found. Using PIL default.[/yellow]")
        return None

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        return tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    def _draw_radial_gradient(self, draw: ImageDraw.Draw, size: Tuple[int, int], 
                            colors: List[Tuple[int, int, int]]):
        """Draw a radial gradient background."""
        w, h = size
        cx, cy = w // 2, h // 2
        max_r = int((w**2 + h**2)**0.5 / 2)
        c1, c2, c3 = colors[:3]
        for r in range(max_r, 0, -1):
            if r > max_r / 2:
                blend = 1 - (r - max_r / 2) / (max_r / 2)
                c = tuple(int(c2[j] * (1 - blend) + c1[j] * blend) for j in range(3))
            else:
                blend = r / (max_r / 2)
                c = tuple(int(c3[j] * (1 - blend) + c2[j] * blend) for j in range(3))
            draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=c)

    def _draw_aura_blobs(self, image: Image.Image, colors: List[Tuple[int, int, int]]):
        """Add atmospheric blob effects."""
        overlay = Image.new("RGBA", image.size)
        draw = ImageDraw.Draw(overlay)
        w, h = image.size
        for _ in range(5):
            radius = random.randint(w // 4, w // 2)
            ox = random.randint(-w // 8, w // 8)
            oy = random.randint(-h // 8, h // 8)
            x, y = w // 2 + ox, h // 2 + oy
            r, g, b = random.choice(colors)
            a = random.randint(40, 70)
            draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], 
                        fill=(r, g, b, a))
        return Image.alpha_composite(
            image.convert("RGBA"), 
            overlay.filter(ImageFilter.GaussianBlur(radius=90))
        ).convert("RGB")

    def _measure_text(self, draw: ImageDraw.Draw, text: str, 
                     font: ImageFont.ImageFont) -> Tuple[int, int]:
        """Measure text dimensions."""
        bbox = draw.textbbox((0, 0), text, font=font, align="center")
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])

    def _wrap_text(self, draw: ImageDraw.Draw, text: str, font: ImageFont.ImageFont, 
                  max_w: int, max_lines: int = 3) -> str:
        """Wrap text to fit within specified width."""
        words = text.split()
        if not words:
            return text
        lines: List[str] = []
        cur: List[str] = []
        for w in words:
            trial = " ".join(cur + [w])
            w_px, _ = self._measure_text(draw, trial, font)
            if w_px <= max_w:
                cur.append(w)
            else:
                if cur:
                    lines.append(" ".join(cur))
                cur = [w]
                if len(lines) >= max_lines - 1:
                    break
        if cur and len(lines) < max_lines:
            lines.append(" ".join(cur))
        return "\n".join(lines) if lines else text

    def _fit_text(self, draw: ImageDraw.Draw, text: str, max_w: int, 
                 max_h: int) -> Tuple[Optional[ImageFont.ImageFont], str]:
        """Find the best font size and wrap text to fit dimensions."""
        text = text.upper()
        for size in range(220, 22, -4):
            try:
                if self.font_path:
                    font = ImageFont.truetype(self.font_path, size)
                else:
                    font = ImageFont.load_default()
                wrapped = self._wrap_text(draw, text, font, max_w)
                w_px, h_px = self._measure_text(draw, wrapped, font)
                if w_px <= max_w and h_px <= max_h:
                    return font, wrapped
            except Exception:
                continue
        try:
            font = ImageFont.truetype(self.font_path, 28) if self.font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        wrapped = self._wrap_text(draw, text, font, max_w)
        return font, wrapped

    def _render_typography(self, image: Image.Image, text: str):
        """Render text with shadow effect."""
        draw = ImageDraw.Draw(image)
        w, h = image.size
        font, wrapped = self._fit_text(draw, text, int(w * 0.84), int(h * 0.34))
        if not font:
            return
        tw, th = self._measure_text(draw, wrapped, font)
        x = (w - tw) / 2
        y = (h - th) * 0.56
        shadow = int(font.size * 0.03)
        draw.text((x + shadow, y + shadow), wrapped, font=font, 
                 fill=(0, 0, 0, 160), align="center")
        draw.text((x, y), wrapped, font=font, fill=(255, 255, 255), align="center")

    def generate(self, name: str, colors_hex: List[str], 
                size: Tuple[int, int] = Config.DEFAULT_IMAGE_SIZE) -> Image.Image:
        """Generate a complete cover art image."""
        colors_rgb = [self._hex_to_rgb(c) for c in colors_hex]
        img = Image.new("RGB", size)
        draw = ImageDraw.Draw(img)
        self._draw_radial_gradient(draw, size, colors_rgb)
        img = self._draw_aura_blobs(img, colors_rgb)
        # Add subtle grain texture
        noise = np.random.normal(0, 10, (size[1], size[0], 3)).astype(np.int16)
        arr = np.array(img, dtype=np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        self._render_typography(img, name)
        return img


def compress_jpeg_under(img: Image.Image, max_bytes: int) -> Optional[bytes]:
    """Compress JPEG image to fit under byte limit."""
    for dim in [640, 600, 560, 520, 480, 440, 400]:
        im = img.copy()
        im = im.resize((dim, dim), Image.Resampling.LANCZOS)
        for quality in [90, 85, 80, 75, 70, 65, 60, 55, 50]:
            buf = io.BytesIO()
            try:
                im.save(buf, "JPEG", quality=quality, optimize=True, 
                       progressive=True, subsampling=2)
            except OSError:
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=quality, progressive=True)
            data = buf.getvalue()
            if len(data) <= max_bytes:
                return data
    return None


# --- Spotify Service ---
class SpotifyService:
    """Handles all Spotify API interactions."""
    
    def __init__(self):
        self.client = spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                client_id=Config.SPOTIFY_CLIENT_ID,
                client_secret=Config.SPOTIFY_CLIENT_SECRET,
                redirect_uri=Config.SPOTIFY_REDIRECT_URI,
                scope="playlist-read-private playlist-modify-public playlist-modify-private ugc-image-upload",
                cache_path=str(Path(".eclyra_cache") / ".spotify_cache"),
                open_browser=True,
            ),
            requests_timeout=30,
            retries=3,
        )
        self.me = self.client.current_user()
        self.audio_features_available = None

    def get_playlist_tracks(self, playlist_id: str) -> List[Song]:
        """Fetch all tracks from a Spotify playlist."""
        songs: List[Song] = []
        fields = "items(track(id,name,artists(name),popularity,album(name,release_date))),next,total"
        results = self.client.playlist_items(
            playlist_id,
            fields=fields,
            market=Config.DEFAULT_MARKET,
            additional_types=("track",),
        )
        total = results.get("total") or 0
        while True:
            for it in results.get("items", []):
                t = it.get("track")
                if not t:
                    continue
                tid = t.get("id")
                if not tid:
                    continue
                release_date = ((t.get("album") or {}).get("release_date") or "").split("-")
                songs.append(Song(
                    id=tid,
                    name=t.get("name", ""),
                    artists=[a.get("name", "") for a in t.get("artists", []) if a],
                    popularity=t.get("popularity"),
                    album_name=(t.get("album") or {}).get("name"),
                    release_year=release_date[0] if release_date else None,
                ))
            if results.get("next"):
                results = self.client.next(results)
            else:
                break
        if total and len(songs) != total:
            console.print(f"[yellow]Note: Retrieved {len(songs)} of reported {total} items.[/yellow]")
        return songs

    @retry_net()
    def create_playlist(self, name: str, desc: str, public: bool = True) -> dict:
        """Create a new Spotify playlist."""
        return self.client.user_playlist_create(
            self.me["id"], name=name[:100], public=public, description=desc[:300]
        )

    @retry_net()
    def add_tracks_in_batches(self, playlist_id: str, track_ids: List[str]):
        """Add tracks to playlist in batches with rate limiting."""
        for i in range(0, len(track_ids), 100):
            batch = track_ids[i:i+100]
            try:
                self.client.playlist_add_items(playlist_id, batch)
            except SpotifyException as e:
                if e.http_status == 429:
                    retry_after = int(getattr(e, "headers", {}).get("Retry-After", "3"))
                    time.sleep(retry_after)
                    self.client.playlist_add_items(playlist_id, batch)
                else:
                    raise

    @retry_net()
    def upload_cover(self, playlist_id: str, b64_jpg: str):
        """Upload cover art to Spotify playlist."""
        self.client.playlist_upload_cover_image(playlist_id, b64_jpg)

    def audio_features(self, track_ids: List[str]) -> Dict[str, dict]:
        """
        Attempt to get audio features with graceful fallback.
        Returns empty dict if not available (post-Nov 2024 Spotify API restrictions).
        """
        if self.audio_features_available is False:
            return {}
        
        if self.audio_features_available is None:
            try:
                test_id = track_ids[:1] if track_ids else []
                if test_id:
                    self.client.audio_features(test_id)
                    self.audio_features_available = True
                    console.print("[green]✓ Audio features available - enabling advanced track smoothing.[/green]")
                else:
                    self.audio_features_available = False
                    return {}
            except SpotifyException as e:
                if e.http_status == 403:
                    self.audio_features_available = False
                    console.print("[yellow]⚠ Audio features not available (Spotify API restriction). Using alternative ordering logic.[/yellow]")
                    return {}
                else:
                    raise

        out: Dict[str, dict] = {}
        try:
            for i in range(0, len(track_ids), 100):
                feats = self.client.audio_features(track_ids[i:i+100]) or []
                for f in feats:
                    if f and f.get("id"):
                        out[f["id"]] = f
        except SpotifyException as e:
            if e.http_status == 403:
                self.audio_features_available = False
                console.print("[yellow]⚠ Audio features access revoked during operation. Continuing without smoothing.[/yellow]")
                return {}
            else:
                raise
        return out


# --- Lyrics Service ---
class LyricsService:
    """Handles lyrics fetching with caching."""
    
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.cache: Dict[str, Optional[str]] = {}
        self._load()
        self.client: Optional[lyricsgenius.Genius] = None

    def _load(self):
        """Load lyrics cache from disk."""
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}

    def _save(self):
        """Save lyrics cache to disk."""
        try:
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), 
                          encoding="utf-8")
            tmp.replace(self.cache_path)
        except Exception:
            pass

    def _ensure_client(self):
        """Initialize Genius client if needed."""
        if self.client is None:
            self.client = lyricsgenius.Genius(
                Config.GENIUS_ACCESS_TOKEN,
                verbose=False,
                remove_section_headers=True,
                skip_non_songs=True,
                timeout=15,
            )

    @staticmethod
    def normalize_title(t: str) -> str:
        """Normalize song title for better search results."""
        t = re.sub(r"\(feat\..*?\)|\[feat\..*?\]", "", t, flags=re.I)
        t = re.sub(r"-\s*.*remaster.*|\(.*version\)|- live.*|\(live.*\)", "", t, flags=re.I)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def get_key(self, s: Song) -> str:
        """Generate cache key for song."""
        artist = " & ".join(s.artists) if s.artists else ""
        return f"{s.name} - {artist}"

    def fetch(self, s: Song) -> Optional[str]:
        """Fetch lyrics for a single song."""
        self._ensure_client()
        name = self.normalize_title(s.name)
        artist = " & ".join(s.artists) if s.artists else ""
        res = self.client.search_song(name, artist)
        if not res:
            return None
        lyrics = re.sub(r"\[.*?\]\n*|\n{3,}", "\n", res.lyrics).strip()
        if len(lyrics) > Config.MAX_LYRICS_LENGTH:
            cut = lyrics[:Config.MAX_LYRICS_LENGTH]
            cut = cut.rsplit(" ", 1)[0] + "…"
            return cut
        return lyrics

    def populate_lyrics(self, songs: List[Song], max_workers: int = 6, enabled: bool = True):
        """Fetch lyrics for multiple songs concurrently."""
        if not enabled:
            return
        to_fetch = []
        for s in songs:
            k = self.get_key(s)
            if k in self.cache:
                s.lyrics = self.cache[k]
            else:
                to_fetch.append(s)
        if not to_fetch:
            return
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(self.fetch, s): s for s in to_fetch}
            for fut in as_completed(futs):
                s = futs[fut]
                k = self.get_key(s)
                try:
                    lyr = fut.result()
                    self.cache[k] = lyr
                    s.lyrics = lyr
                except Exception:
                    pass
        self._save()


# --- Gemini Integration ---
class GeminiService:
    """Handles AI playlist generation using Google Gemini."""
    
    def __init__(self):
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel("gemini-1.5-flash-latest")

    @retry_net()
    def generate_plan(self, theme: str, songs_meta: List[Dict]) -> PlaylistPlan:
        """Generate playlist plan using AI."""
        prompt = f"""
You are Eclyra, an AI expert in music and storytelling. Create a curated playlist plan from the songs based on the theme.

Rules:
- Produce exactly 5 stages in "story_arc". Each stage has "name", "description", and "sorted_song_ids".
- Assign every provided song id to exactly ONE stage.
- Within each stage, order song ids to make musical sense based on the song information provided.
- Consider factors like song popularity, release year, artist style, and emotional progression.
- Provide 5 "playlist_names".
- Provide a concise "playlist_description".
- Output JSON ONLY, no markdown or code fences.

THEME: "{theme}"

SONGS (array of objects with id, name, artists, popularity, album_name, release_year):
{json.dumps(songs_meta, ensure_ascii=False)}
"""
        resp = self.model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
            request_options={"timeout": 180},
        )
        text = getattr(resp, "text", None)
        if not text:
            try:
                cand = resp.candidates[0]
                parts = getattr(getattr(cand, "content", None), "parts", []) or []
                text = "".join(getattr(p, "text", "") for p in parts)
            except Exception:
                pass
        if not text:
            raise RuntimeError("Empty response from Gemini")
        
        m = re.search(r"\{.*\}\s*$", text, re.S)
        raw = m.group(0) if m else text
        data = json.loads(raw)
        try:
            return PlaylistPlan.model_validate(data)
        except ValidationError as ve:
            raise RuntimeError(f"AI response validation failed: {ve}") from ve


# --- Utility Functions ---
def extract_playlist_id(url_or_id: str) -> Optional[str]:
    """Extract Spotify playlist ID from URL or return ID if already provided."""
    s = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9]{22}", s):
        return s
    
    m = re.match(r"spotify:playlist:([A-Za-z0-9]{22})", s)
    if m:
        return m.group(1)
    
    try:
        u = up.urlparse(s)
        if "spotify.com" in (u.netloc or ""):
            parts = [p for p in (u.path or "").split("/") if p]
            if "playlist" in parts:
                idx = parts.index("playlist")
                if idx + 1 < len(parts):
                    pid = parts[idx + 1]
                    pid = pid.split("?")[0]
                    if re.fullmatch(r"[A-Za-z0-9]{22}", pid):
                        return pid
    except Exception:
        pass
    return None


def smart_order_by_metadata(stage_ids: List[str], songs: List[Song]) -> List[str]:
    """Alternative ordering logic when audio features aren't available."""
    if len(stage_ids) < 2:
        return stage_ids
    
    song_map = {s.id: s for s in songs}
    stage_songs = [song_map[sid] for sid in stage_ids if sid in song_map]
    
    if not stage_songs:
        return stage_ids
    
    def sort_key(song: Song):
        popularity = song.popularity or 0
        try:
            year = int(song.release_year) if song.release_year else 2000
        except (ValueError, TypeError):
            year = 2000
        name = song.name.lower()
        return (-popularity, -year, name)
    
    sorted_songs = sorted(stage_songs, key=sort_key)
    return [s.id for s in sorted_songs]


def smooth_order_by_features(stage_ids: List[str], feats: Dict[str, dict]) -> List[str]:
    """Audio features based ordering - only used when features are available."""
    if len(stage_ids) < 3:
        return stage_ids
    
    keys = ["tempo", "energy", "valence", "danceability", "acousticness"]
    vals: Dict[str, List[float]] = {k: [] for k in keys}
    
    for sid in stage_ids:
        f = feats.get(sid)
        if not f:
            continue
        for k in keys:
            v = f.get(k)
            if v is not None:
                vals[k].append(float(v))
    
    mins = {k: min(vals[k]) if vals[k] else 0.0 for k in keys}
    maxs = {k: max(vals[k]) if vals[k] else 1.0 if mins[k] == 0 else mins[k] + 1.0 for k in keys}

    def norm(sid: str) -> np.ndarray:
        f = feats.get(sid)
        if not f:
            return np.zeros(len(keys), dtype=float)
        arr = []
        for k in keys:
            v = float(f.get(k, mins[k]))
            lo, hi = mins[k], maxs[k]
            arr.append((v - lo) / (hi - lo) if hi > lo else 0.0)
        return np.array(arr, dtype=float)

    remaining = list(stage_ids)
    remaining.sort(key=lambda x: feats.get(x, {}).get("tempo", 120.0))
    start = remaining[len(remaining)//2]
    order = [start]
    remaining.remove(start)
    cur = start
    
    while remaining:
        v = norm(cur)
        nxt = min(remaining, key=lambda x: np.linalg.norm(v - norm(x)))
        order.append(nxt)
        remaining.remove(nxt)
        cur = nxt
    return order


def repair_and_finalize_plan(plan: PlaylistPlan, songs: List[Song], 
                           feats: Dict[str, dict], enable_smoothing: bool = True) -> Tuple[List[StageModel], List[Song]]:
    """Repair AI plan and finalize song ordering."""
    id_to_song = {s.id: s for s in songs}
    provided_ids = set(id_to_song.keys())

    # Ensure 5 stages exist
    if not plan.story_arc or len(plan.story_arc) != 5:
        needed = 5
        plan.story_arc = plan.story_arc[:5] if plan.story_arc else []
        while len(plan.story_arc) < needed:
            plan.story_arc.append(StageModel(
                name=f"Stage {len(plan.story_arc)+1}", 
                description="", 
                sorted_song_ids=[]
            ))

    # Keep only valid ids
    for st in plan.story_arc:
        st.sorted_song_ids = [sid for sid in (st.sorted_song_ids or []) if sid in provided_ids]

    assigned = set(sid for st in plan.story_arc for sid in st.sorted_song_ids)
    missing = [sid for sid in provided_ids if sid not in assigned]

    # Distribute missing songs
    for i, sid in enumerate(missing):
        plan.story_arc[i % len(plan.story_arc)].sorted_song_ids.append(sid)

    # Deduplicate
    seen: set = set()
    for st in plan.story_arc:
        unique = []
        for sid in st.sorted_song_ids:
            if sid not in seen:
                unique.append(sid)
                seen.add(sid)
        st.sorted_song_ids = unique

    # Smart ordering based on available data
    if enable_smoothing:
        for st in plan.story_arc:
            if feats:
                st.sorted_song_ids = smooth_order_by_features(st.sorted_song_ids, feats)
            else:
                st.sorted_song_ids = smart_order_by_metadata(st.sorted_song_ids, songs)

    final_order: List[Song] = []
    for st in plan.story_arc:
        final_order.extend([id_to_song[sid] for sid in st.sorted_song_ids if sid in id_to_song])

    # Ensure all songs are included
    final_ids = {s.id for s in final_order}
    tail = [s for s in songs if s.id not in final_ids]
    final_order.extend(tail)
    return plan.story_arc, final_order


# --- CLI Application ---
class EclyraCLI:
    """Main CLI application class."""
    
    def __init__(self, gemini: GeminiService, no_lyrics: bool = False, no_cover: bool = False):
        self.sp: Optional[SpotifyService] = None
        self.gemini = gemini
        self.no_lyrics = no_lyrics
        self.no_cover = no_cover
        self.cache_dir = Path(".eclyra_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.lyrics = LyricsService(self.cache_dir / "lyrics_cache.json")
        self.session_id = str(int(time.time()))
        self.cover_generator = CreativeCoverArtGenerator()

    def _display_header(self):
        """Display application header."""
        title = Text("Eclyra", style="bold magenta", justify="center")
        tagline = Text("Turn Memories into Mixtapes", style="italic", justify="center")
        header_text = Text.assemble(title, "\n", tagline, justify="center")
        console.print(Panel(header_text, border_style="magenta", expand=False, padding=(1, 4)))

    def _display_step(self, number: int, title: str):
        """Display step header."""
        console.rule(f"[bold]Step {number}: {title}[/bold]", style="cyan", characters="─")

    def _get_progress_bar(self) -> Progress:
        """Create progress bar."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        )

    def initialize_spotify(self):
        """Initialize Spotify service."""
        self.sp = SpotifyService()

    def extract_playlist_id(self, url: str) -> Optional[str]:
        """Extract playlist ID from URL."""
        return extract_playlist_id(url)

    def get_playlist_tracks(self, playlist_id: str) -> List[Song]:
        """Fetch playlist tracks."""
        self._display_step(1, "Fetching Your Music")
        try:
            with self._get_progress_bar() as progress:
                task = progress.add_task("[cyan]Fetching tracks...", total=None)
                songs = self.sp.get_playlist_tracks(playlist_id)
                progress.update(task, completed=100)
            console.print(f"[green]✓ Found {len(songs)} songs.[/green]")
            return songs
        except SpotifyException as e:
            console.print(f"[red]Spotify API error: {e}[/red]")
        except Exception as e:
            console.print(f"[red]Unexpected error while fetching tracks: {e}[/red]")
        return []

    def get_lyrics(self, songs: List[Song]):
        """Fetch lyrics for songs."""
        self._display_step(2, "Analyzing Lyrics")
        with self._get_progress_bar() as progress:
            task = progress.add_task("[cyan]Fetching lyrics...", total=len(songs))
            self.lyrics.populate_lyrics(songs, max_workers=6, enabled=(not self.no_lyrics))
            progress.update(task, completed=len(songs))

    def generate_playlist_from_theme(self, theme: str, songs: List[Song]) -> Optional[PlaylistPlan]:
        """Generate playlist plan from theme."""
        self._display_step(3, "Crafting Your Playlist Story")
        song_meta = [s.to_dict(include_lyrics=False) for s in songs]
        try:
            with console.status("[bold yellow]AI is crafting your playlist...[/bold yellow]"):
                plan = self.gemini.generate_plan(theme, song_meta)
            console.print("[green]✓ AI has finished crafting the playlist.[/green]")
            return plan
        except Exception as e:
            console.print(f"[red]Error during AI playlist generation: {e}[/red]")
            return None

    def create_playlist_on_spotify(self, name: str, desc: str, tracks: List[Song]) -> Optional[str]:
        """Create playlist on Spotify."""
        self._display_step(4, "Saving to Spotify")
        if not tracks:
            console.print("[red]No songs to add.[/red]")
            return None
        try:
            playlist = self.sp.create_playlist(name=name, desc=desc, public=True)
            console.print(f"✓ Playlist '{name}' created on Spotify.")
            uris = [f"spotify:track:{t.id}" for t in tracks]
            self.sp.add_tracks_in_batches(playlist["id"], uris)
            console.print(f"✓ Added {len(tracks)} songs.")
            
            if not self.no_cover and Confirm.ask("\n[bold]Generate and upload a unique cover art?[/bold]", default=True):
                colors = random.choice(PREDEFINED_PALETTES)
                img = self.cover_generator.generate(name, colors)
                jpg = compress_jpeg_under(img, Config.MAX_IMAGE_SIZE_BYTES)
                if not jpg:
                    console.print("[red]Error: Could not compress image under 256 KB.[/red]")
                else:
                    b64 = base64.b64encode(jpg).decode("utf-8")
                    self.sp.upload_cover(playlist["id"], b64)
                    console.print("[green]✓ Cover art uploaded successfully.[/green]")
            return playlist.get("external_urls", {}).get("spotify")
        except Exception as e:
            console.print(f"[bold red]Failed to create playlist: {e}[/bold red]")
            return None

    def run(self):
        """Run the main application."""
        self._display_header()
        
        # Initialize Spotify
        try:
            console.print("Ready to begin? First, let's connect to Spotify.", justify="center")
            self.initialize_spotify()
            console.print("[green]✓ Connected to Spotify.[/green]")
        except Exception as e:
            console.print(f"[red]Fatal Error: Could not authenticate with Spotify.[/red]\n[dim]{e}[/dim]")
            return

        # Get playlist input
        pid_in = Prompt.ask("\n🔗 Enter a Spotify Playlist URL or ID")
        playlist_id = self.extract_playlist_id(pid_in)
        if not playlist_id:
            console.print("[red]Invalid URL/ID format. Exiting.[/red]")
            return

        songs = self.get_playlist_tracks(playlist_id)
        if not songs:
            console.print("[red]No songs found or playlist is invalid. Exiting.[/red]")
            return

        # Optional lyrics
        if not self.no_lyrics and Confirm.ask("Fetch lyrics to enrich context? (slower)", default=False):
            self.get_lyrics(songs)
        else:
            self.no_lyrics = True

        theme = Prompt.ask("✨ Now, describe the story, feeling, or theme for your playlist")

        plan = self.generate_playlist_from_theme(theme, songs)
        if not plan:
            console.print("[red]Could not generate playlist data from AI. Exiting.[/red]")
            return

        # Check audio features availability
        console.print("[dim]Checking audio features availability...[/dim]")
        feats = self.sp.audio_features([s.id for s in songs])

        # Finalize plan
        story_arc, final_track_order = repair_and_finalize_plan(plan, songs, feats, enable_smoothing=True)

        # Display story arc
        table = Table(title="[bold]Your Playlist's Generated Story Arc[/bold]", 
                     title_style="magenta", show_header=True, header_style="blue")
        table.add_column("Stage", style="cyan", no_wrap=True)
        table.add_column("Description")
        table.add_column("Songs")

        song_map = {s.id: s for s in songs}
        for st in story_arc:
            names = [f"- {song_map[sid].name}" for sid in st.sorted_song_ids if sid in song_map]
            table.add_row(st.name, st.description, "\n".join(names))
        console.print(table)

        # Choose playlist name
        name_options = plan.playlist_names or [f"{theme} — Mix"]
        playlist_desc = plan.playlist_description or f"A playlist about {theme}."
        name_table = Table(title="Choose a Title for Your Playlist", 
                          show_header=True, header_style="blue")
        name_table.add_column("#", style="cyan")
        name_table.add_column("Playlist Name")
        name_table.add_row("[bold]0[/bold]", "[dim]Quit[/dim]")
        for i, name in enumerate(name_options):
            name_table.add_row(str(i + 1), name)
        console.print(name_table)
        
        choices = ["0"] + [str(i + 1) for i in range(len(name_options))]
        choice = IntPrompt.ask("Enter the number of your choice", choices=choices, show_choices=False)
        if choice == 0:
            console.print("[yellow]Aborted by user.[/yellow]")
            return
        playlist_name = name_options[choice - 1]

        console.rule("[bold]Finalization[/bold]", style="cyan")
        console.print(f"You chose: [bold magenta]{playlist_name}[/bold magenta]")
        console.print(f"[dim]Description: '{playlist_desc}'[/dim]")

        if Confirm.ask(f"\nCreate playlist '{playlist_name}' on Spotify?", default=True):
            url = self.create_playlist_on_spotify(playlist_name, playlist_desc, final_track_order)
            if url:
                console.print(Panel(
                    Text(f"✨ Success! ✨\n\nYour new playlist is ready.\n\n🔗 {url}", 
                         justify="center"), 
                    title="Showcase Ready", border_style="green", expand=False
                ))
            else:
                console.print("[red]Playlist creation failed.[/red]")
        else:
            console.print("[yellow]Aborted by user.[/yellow]")


def main():
    """Application entry point."""
    try:
        # Validate configuration
        missing_keys = []
        if Config.SPOTIFY_CLIENT_ID.startswith("your_"):
            missing_keys.append("SPOTIFY_CLIENT_ID")
        if Config.SPOTIFY_CLIENT_SECRET.startswith("your_"):
            missing_keys.append("SPOTIFY_CLIENT_SECRET")
        if Config.GENIUS_ACCESS_TOKEN.startswith("your_"):
            missing_keys.append("GENIUS_ACCESS_TOKEN")
        if Config.GEMINI_API_KEY.startswith("your_"):
            missing_keys.append("GEMINI_API_KEY")
        
        if missing_keys:
            console.print("[red]Missing API keys. Please set the following environment variables or update the Config class:[/red]")
            for key in missing_keys:
                console.print(f"[red]- {key}[/red]")
            console.print("\n[dim]See README.md for setup instructions.[/dim]")
            sys.exit(1)
        
        gemini = GeminiService()
        try:
            if getattr(gemini.model, "_client", None):
                atexit.register(lambda: gemini.model._client.close())
        except Exception:
            pass
        
        no_lyrics = "--no-lyrics" in sys.argv
        no_cover = "--no-cover" in sys.argv
        cli = EclyraCLI(gemini, no_lyrics=no_lyrics, no_cover=no_cover)
        cli.run()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Process interrupted by user. Exiting.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]An unexpected fatal error occurred: {e}[/bold red]")
        console.print_exception(show_locals=False)
        sys.exit(1)


if __name__ == "__main__":
    main()
