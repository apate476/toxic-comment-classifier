# Data Directory

Store all data files for the toxic_comment_classifier project here.

## Source

Jigsaw Toxic Comment Classification Challenge —
[kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge).
Mirrored to a shared Google Drive folder and versioned with DVC:
[Drive folder](https://drive.google.com/drive/folders/1Va1f8JbuGsPH9tGZCZOsXk8rHgcisZNr?usp=sharing).

## Structure

- **`raw/`** — Original, immutable data as received. Never modify files here.
  - `train.csv` — 159,571 rows: `id`, `comment_text`, and 6 binary labels (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`).
  - `test.csv` — `id`, `comment_text`.
  - `test_labels.csv` — `id` + the 6 label columns.
- **`processed/`** — Cleaned, transformed, and feature-engineered data ready for modeling. Produced by `python -m toxic_comment_classifier.data.make_dataset` (or `make data`).

## Pulling the data (DVC + Google Drive)

The `gdrive` remote uses OAuth, so every collaborator must register their own
client ID and secret before running `dvc pull`.

1. **Create your own OAuth client** in the Google Cloud Console:
   - New project → **APIs & Services → Library** → enable **Google Drive API**.
   - **APIs & Services → Credentials** → **+ CREATE CREDENTIALS → OAuth client ID**.
   - Application type: **Desktop app** (not Web application — loopback redirects only work for Desktop clients).
   - Copy the client ID and client secret.
2. **Add yourself as a test user** (required while the OAuth consent screen is in "Testing" status):
   - **APIs & Services → OAuth consent screen → Test users → + ADD USERS** → add the Google account you'll authorize with.
   - That account also needs access to the shared Drive folder linked above.
3. **Register your credentials locally** (writes to the gitignored `.dvc/config.local`):
   ```bash
   dvc remote modify --local gdrive gdrive_client_id     'YOUR_CLIENT_ID'
   dvc remote modify --local gdrive gdrive_client_secret 'YOUR_CLIENT_SECRET'
   ```
4. **Pull**:
   ```bash
   dvc pull
   ```
   The first run opens a browser. Click through the "Google hasn't verified this app" warning (Advanced → Go to ...) and grant Drive access. Subsequent pulls reuse the cached token in `~/Library/Caches/pydrive2fs/`.

After this, `data/raw/` will contain the three CSVs above.

### Common errors

- `redirect_uri ... doesn't comply with Google's OAuth 2.0 policy` → OAuth client was created as **Web application**. Recreate it as **Desktop app**.
- `Error 403: access_denied` → The signing-in Google account isn't listed under **Test users** on the OAuth consent screen.
- `Google Drive API has not been used in project ... or it is disabled` → Enable the Drive API in the Cloud project that owns your OAuth client (the linked URL in the error message goes straight to the right page).

## Loading data in code

```python
from toxic_comment_classifier.data import load_raw, load_processed, save_processed

df = load_raw("train.csv")           # reads data/raw/train.csv
processed = load_processed("...csv") # reads data/processed/...csv
```

The raw → processed pipeline entrypoint is `python -m toxic_comment_classifier.data.make_dataset` (also exposed as `make data`).

## Best Practices

- Use **DVC** to version large data files instead of Git
- Track `.dvc` files in Git; store actual data remotely
- **Never commit** large data files directly to Git
- Document data sources and transformations in notebooks or scripts
