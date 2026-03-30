# 🏘️ NYC Gentrification Early Warning System

An end-to-end neural network project that uses an LSTM to score NYC neighborhoods by gentrification risk — deployable as a live web app.

**Stack:** PyTorch → FastAPI → Next.js → Vercel + Railway

---

## Architecture

```
NYC Open Data / Zillow / ACS Census
        ↓
  data/fetch_data.py       ← pulls raw data
        ↓
  data/preprocess.py       ← feature engineering + sequence building
        ↓
  model/train.py           ← trains LSTM, saves checkpoint
        ↓
  api/main.py              ← FastAPI serves predictions  (Railway)
        ↓
  frontend/                ← Next.js interactive map     (Vercel)
```

---

## Project Structure

```
nyc-gentrification-watch/
├── data/
│   ├── fetch_data.py        # NYC Open Data + Zillow fetchers
│   ├── preprocess.py        # Feature engineering → numpy sequences
│   ├── raw/                 # Raw CSVs (gitignored)
│   └── processed/           # X.npy, y.npy, meta.csv
├── model/
│   ├── lstm_model.py        # PyTorch LSTM + attention
│   ├── train.py             # Training loop + early stopping
│   └── checkpoints/         # best_model.pt, model_meta.json
├── api/
│   ├── main.py              # FastAPI app
│   ├── requirements.txt
│   └── railway.toml         # Railway deploy config
├── frontend/
│   ├── src/
│   │   ├── pages/index.tsx  # Main dashboard
│   │   ├── components/MapView.tsx
│   │   └── styles/globals.css
│   ├── package.json
│   ├── next.config.js
│   └── vercel.json
└── README.md
```

---

## Setup & Run Locally

### 1. Data Pipeline

```bash
cd data
pip install pandas numpy requests
python fetch_data.py       # generates synthetic rent data
python preprocess.py       # builds X.npy, y.npy
```

To use **real** NYC Open Data:
1. Get a free token at [data.cityofnewyork.us](https://data.cityofnewyork.us)
2. `export NYC_OPEN_DATA_TOKEN=your_token`
3. Uncomment the fetchers in `fetch_data.py`

For **real rent data**: Download ZORI CSV from [Zillow Research Data](https://www.zillow.com/research/data/) and replace `data/raw/rent_by_neighborhood.csv`.

### 2. Train Model

```bash
cd model
pip install torch numpy pandas
python train.py
# → saves model/checkpoints/best_model.pt
```

### 3. Run API

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

### 4. Run Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

---

## Model Details

### Architecture
- **LSTM** with attention pooling (2 layers, 64 hidden units)
- **Input**: 12-month sequence × 8 features per timestep
- **Output**: gentrification risk score (0–1)
- **Training**: BCE loss with class-weighting, AdamW, early stopping

### Features (per timestep)
| Feature | Description |
|---|---|
| `rent_yoy` | Year-over-year rent growth |
| `rent_3m_momentum` | 3-month rolling rent momentum |
| `rent_vs_nyc` | Rent relative to NYC median |
| `permit_intensity` | New construction/renovation permits per 1k residents |
| `new_licenses` | New bar/restaurant license applications |
| `housing_complaints` | 311 housing maintenance complaints (inverse) |
| `demo_shift` | Demographic shift index (Census) |
| `income_index` | Median income relative to borough |

### Label
Binary: `1` if rent YoY growth > 8% in the next 6 months.

---

## Deploy

### Backend → Railway

1. Push repo to GitHub
2. Create new Railway project → "Deploy from GitHub"
3. Set root directory to `api/`
4. Railway auto-detects `railway.toml` and `requirements.txt`
5. Note your Railway URL (e.g. `https://your-api.railway.app`)

### Frontend → Vercel

1. Import repo to [vercel.com](https://vercel.com)
2. Set root directory to `frontend/`
3. Add env var: `NEXT_PUBLIC_API_URL=https://your-api.railway.app`
4. Deploy → live at `https://your-app.vercel.app`

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | API info |
| `GET` | `/health` | Health check |
| `GET` | `/neighborhoods` | All neighborhoods + risk scores |
| `GET` | `/neighborhoods/{name}` | Detailed neighborhood data |
| `POST` | `/predict` | Live LSTM inference |
| `GET` | `/boroughs/{borough}` | Filter by borough |

---

## Who this is for

- **Tenant advocacy orgs** (ANHD, Tenants & Neighbors): prioritize outreach to high-risk areas *before* displacement happens
- **Community boards**: justify affordable housing protections with data
- **Urban researchers**: track early gentrification signals over time

---

## Data Sources

- [NYC Open Data — DOB Permits](https://data.cityofnewyork.us/Housing-Development/DOB-Permit-Issuance/ipu4-2q9a)
- [NYC Open Data — 311 Complaints](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9)
- [NY State — SLA Liquor Licenses](https://data.ny.gov/Economic-Development/Liquor-Authority-Quarterly-List-of-Active-Licenses/hrvs-fxs2)
- [Zillow Research — ZORI Rent Data](https://www.zillow.com/research/data/)
- [US Census Bureau — ACS 5-Year Estimates](https://www.census.gov/data/developers/data-sets/acs-5year.html)

---

Built with PyTorch · FastAPI · Next.js · Leaflet · Recharts
