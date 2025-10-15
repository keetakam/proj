import json
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import math

# helper: parse possible coord formats -> (lat, lon) or (None, None)
def parse_coords(val):
    try:
        if val is None:
            return (None, None)

        # list/tuple like [lat, lon] or ['15.2','100.3']
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            try:
                lat = float(val[0])
                lon = float(val[1])
            except Exception:
                return (None, None)
            if not (math.isfinite(lat) and math.isfinite(lon)):
                return (None, None)
            return (lat, lon)

        # string "lat,lon" or "lat lon"
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return (None, None)
            parts = [p for p in re_split_coord(s) if p]
            if len(parts) >= 2:
                try:
                    lat = float(parts[0].replace(',', '.'))
                    lon = float(parts[1].replace(',', '.'))
                except Exception:
                    return (None, None)
                if not (math.isfinite(lat) and math.isfinite(lon)):
                    return (None, None)
                return (lat, lon)
        return (None, None)
    except Exception:
        return (None, None)

# small helper to split coord string
def re_split_coord(s):
    import re
    return re.split(r'[,\s]+', s.strip())

# helper: ตรวจว่าแถวมีพิกัดหรือไม่
def row_has_coord(row):
    # check 'สาขา' branches
    if 'สาขา' in row.index and row['สาขา']:
        try:
            for b in (row['สาขา'] or []):
                coord_raw = None
                if isinstance(b, dict):
                    coord_raw = b.get('พิกัด')
                elif isinstance(b, (list, tuple)) and len(b) > 2:
                    coord_raw = b[2]
                lat, lon = parse_coords(coord_raw)
                if lat is not None and lon is not None:
                    return True
        except Exception:
            pass
    # fallback check lat/lon columns
    lat_candidates = [c for c in row.index if c and 'lat' in c.lower()]
    lon_candidates = [c for c in row.index if c and ('lon' in c.lower() or 'lng' in c.lower())]
    if lat_candidates and lon_candidates:
        try:
            lat = row[lat_candidates[0]]
            lon = row[lon_candidates[0]]
            lat_f, lon_f = parse_coords([lat, lon]) if not (isinstance(lat, (int, float)) and isinstance(lon, (int, float))) else (float(lat), float(lon))
            if lat_f is not None and lon_f is not None:
                return True
        except Exception:
            pass
    return False

# helper: สร้าง markers จาก DataFrame
def build_markers(df):
    markers = []
    # 1) ใช้คอลัมน์ 'สาขา'
    if 'สาขา' in df.columns:
        for _, row in df.iterrows():
            company = row.get('ชื่อบริษัท') or ''
            license_no = row.get('เลขที่ใบอนุญาต') or ''
            branches = row.get('สาขา') or []
            if isinstance(branches, list):
                for b in branches:
                    addr = None
                    phone = None
                    coord_raw = None
                    if isinstance(b, dict):
                        addr = b.get('ที่อยู่')
                        phone = b.get('โทรศัพท์')
                        coord_raw = b.get('พิกัด')
                    elif isinstance(b, (list, tuple)) and len(b) >= 3:
                        addr = b[0]
                        phone = b[1]
                        coord_raw = b[2]
                    lat, lon = parse_coords(coord_raw)
                    if lat is not None and lon is not None:
                        popup = f"{company}\n{addr or ''}\n{phone or ''}\n{license_no}"
                        markers.append({'lat': lat, 'lon': lon, 'popup': popup})
    # 2) fallback ใช้คอลัมน์ lat/lon
    if not markers:
        lat_cols = [c for c in df.columns if c and 'lat' in c.lower()]
        lon_cols = [c for c in df.columns if c and ('lon' in c.lower() or 'lng' in c.lower())]
        lat_col = lat_cols[0] if lat_cols else None
        lon_col = lon_cols[0] if lon_cols else None
        if lat_col and lon_col:
            for _, row in df.iterrows():
                try:
                    lat_raw = row.get(lat_col)
                    lon_raw = row.get(lon_col)
                    lat, lon = parse_coords(lat_raw) if not (isinstance(lat_raw, (int, float)) and isinstance(lon_raw, (int, float))) else (float(lat_raw), float(lon_raw))
                except Exception:
                    lat, lon = (None, None)
                if lat is not None and lon is not None:
                    popup = f"{row.get('ชื่อบริษัท') or ''}\n{row.get('เลขที่ใบอนุญาต') or ''}"
                    markers.append({'lat': lat, 'lon': lon, 'popup': popup})
    return markers

# Load data
with open('data/merged_pico_companies.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# --- ตัวกรองทั้งหมดในบรรทัดเดียวกัน ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    regions = ['ทั้งหมด'] + sorted(df['ภูมิภาค'].dropna().unique().tolist()) if 'ภูมิภาค' in df.columns else ['ทั้งหมด']
    selected_region = st.selectbox('ภูมิภาค', regions, index=0)

with col2:
    prov_candidates = df['จังหวัด'].dropna().unique().tolist() if 'จังหวัด' in df.columns else []
    if selected_region != 'ทั้งหมด' and 'ภูมิภาค' in df.columns:
        prov_candidates = df.loc[df['ภูมิภาค'] == selected_region, 'จังหวัด'].dropna().unique().tolist()
    provinces = ['ทั้งหมด'] + sorted(prov_candidates)
    selected_province = st.selectbox('จังหวัด', provinces, index=0)

with col3:
    name_query = st.text_input('ชื่อบริษัท', '')

with col4:
    coords_only = st.checkbox('มีพิกัด', value=False)

# apply filters
filtered_df = df.copy()
if selected_region != 'ทั้งหมด':
    filtered_df = filtered_df[filtered_df['ภูมิภาค'] == selected_region]
if selected_province != 'ทั้งหมด':
    filtered_df = filtered_df[filtered_df['จังหวัด'] == selected_province]
if name_query and 'ชื่อบริษัท' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['ชื่อบริษัท'].astype(str).str.contains(name_query, case=False, na=False)]
if coords_only:
    filtered_df = filtered_df[filtered_df.apply(row_has_coord, axis=1)]
filtered_df = filtered_df.reset_index(drop=True)

markers = build_markers(filtered_df)
st.write("แสดงจุด:", len(markers))

# Create map centered on median of markers or Bangkok fallback
center_lat, center_lon = 13.736, 100.518
if markers:
    center_lat = float(markers[0]['lat'])
    center_lon = float(markers[0]['lon'])

m = folium.Map(location=[center_lat, center_lon], zoom_start=7)

# Add markers (cluster if many)
from folium.plugins import MarkerCluster
mc = MarkerCluster().add_to(m)
for mrec in markers:
    lat = mrec.get('lat')
    lon = mrec.get('lon')

    # skip if missing / non-numeric / non-finite
    if lat is None or lon is None:
        continue
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except Exception:
        continue
    if not (math.isfinite(lat_f) and math.isfinite(lon_f)):
        continue

    folium.Marker(
        location=[lat_f, lon_f],
        popup=folium.Popup(mrec.get('popup', ''), max_width=300),
        tooltip=(mrec.get('popup','').splitlines()[0] if mrec.get('popup') else None)
    ).add_to(mc)

# Render map
st_folium(m, width=900, height=600)