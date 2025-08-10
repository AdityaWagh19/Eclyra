# Eclyra — Emotional Playlist Storyteller 🎶

Transform your Spotify playlists into emotional journeys with AI-powered storytelling, intelligent song ordering, and unique cover art.

Requires: Python 3.9+  
License: MIT

---

## 1. FEATURES

- **AI-Powered Storytelling** — Uses Google Gemini to create 5-stage emotional arcs
- **Custom Cover Art** — Generates unique album covers with gradients and typography
- **Smart Song Ordering** — Reorders tracks using audio features or metadata
- **Lyrics Integration** — Fetches and caches lyrics from Genius
- **Spotify Integration** — Creates playlists directly in your Spotify account
- **Interactive CLI** — Rich terminal interface with progress bars

---

## 2. DEMO

Example session:

**Enter a Spotify Playlist URL or ID:**  
`https://open.spotify.com/playlist/...`

**Describe the story, feeling, or theme for your playlist:**  
`A journey through young love`

---

## 3. INSTALLATION

### Prerequisites:
- Python 3.9 or higher
- Spotify Developer Account
- Google AI Studio Account (for Gemini API)
- Genius Developer Account

### Steps:

1. **Clone the repository:**
```bash
git clone https://github.com/AdityaWagh19/Eclyra.git
cd Eclyra
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure API keys:**

**macOS / Linux:**
```bash
export SPOTIFY_CLIENT_ID="your_spotify_client_id"
export SPOTIFY_CLIENT_SECRET="your_spotify_client_secret"
export GENIUS_ACCESS_TOKEN="your_genius_access_token"
export GEMINI_API_KEY="your_gemini_api_key"
```

**Windows PowerShell:**
```powershell
setx SPOTIFY_CLIENT_ID "your_spotify_client_id"
setx SPOTIFY_CLIENT_SECRET "your_spotify_client_secret"
setx GENIUS_ACCESS_TOKEN "your_genius_access_token"
setx GEMINI_API_KEY "your_gemini_api_key"
```

4. **Run Eclyra:**
```bash
python eclyra.py
```

---

## 4. API SETUP

**Spotify API:**
1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new app
3. Note your `Client ID` and `Client Secret`
4. Add `http://127.0.0.1:8080/callback` to Redirect URIs

**Google Gemini API:**
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create an API key

**Genius API:**
1. Go to [Genius API](https://genius.com/api-clients)
2. Create a new API client
3. Note your `Access Token`

---

## 5. USAGE

**Basic:**
```bash
python eclyra.py
```

**Options:**
```bash
python eclyra.py --no-lyrics   # Skip lyrics fetching
python eclyra.py --no-cover    # Skip cover art generation
```

**Interactive Flow:**
1. Connect to Spotify  
2. Input Playlist  
3. Choose Options  
4. Describe Theme  
5. Review Story Arc  
6. Select Playlist Name  
7. Create Playlist with optional cover art

---

## 6. TECHNICAL DETAILS

**Audio Features Handling:**
- Detects availability of audio features at runtime
- Advanced Mode: uses tempo, energy, valence
- Fallback Mode: uses popularity, release year
- Graceful degradation if features are unavailable

**Architecture:**
- **SpotifyService** — Spotify API integration
- **GeminiService** — AI-based playlist creation
- **LyricsService** — Concurrent lyrics fetching with caching
- **CreativeCoverArtGenerator** — Procedural cover art generation
- **EclyraCLI** — Command-line interface

---

## 7. CONTRIBUTING

1. Fork the repository  
2. Create a feature branch  
3. Make your changes  
4. Submit a pull request

---

## 8. LICENSE

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 9. ACKNOWLEDGMENTS

- Spotify — music API  
- Google — Gemini AI  
- Genius — lyrics data  
- Rich — terminal interface library

---

## 10. SUPPORT

Issues and feedback: [GitHub Issues](https://github.com/AdityaWagh19/Eclyra/issues)

---

**Made with ❤️ by Aditya Wagh**  
*Turn your memories into mixtapes with Eclyra*
