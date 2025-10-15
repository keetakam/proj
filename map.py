import json
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import math
import re
import io

# ==================== การตั้งค่า ====================
st.set_page_config(
    page_title="แผนที่บริษัทผลิตภัณฑ์",
    page_icon="📍",
    layout="wide"
)

# ==================== ฟังก์ชันช่วยเหลือ ====================

@st.cache_data
def โหลดข้อมูล(ไฟล์_path):
    """โหลดและเก็บข้อมูล JSON ในแคช"""
    with open(ไฟล์_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)


def แปลงพิกัด(val):
    """แปลงรูปแบบพิกัดต่างๆ -> (lat, lon) หรือ (None, None)"""
    try:
        if val is None:
            return (None, None)

        # รองรับ list/tuple: [lat, lon]
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            try:
                lat, lon = float(val[0]), float(val[1])
                if math.isfinite(lat) and math.isfinite(lon):
                    return (lat, lon)
            except (ValueError, TypeError):
                return (None, None)

        # รองรับ string: "lat,lon" หรือ "lat lon"
        if isinstance(val, str):
            s = val.strip()
            if s:
                parts = re.split(r'[,\s]+', s)
                if len(parts) >= 2:
                    try:
                        lat = float(parts[0].replace(',', '.'))
                        lon = float(parts[1].replace(',', '.'))
                        if math.isfinite(lat) and math.isfinite(lon):
                            return (lat, lon)
                    except (ValueError, TypeError):
                        pass
        
        return (None, None)
    except Exception:
        return (None, None)


def แถวมีพิกัด(row):
    """ตรวจสอบว่าแถวมีพิกัดที่ถูกต้องหรือไม่"""
    # ตรวจสอบคอลัมน์ 'สาขา'
    if 'สาขา' in row.index and row['สาขา']:
        try:
            branches = row['สาขา'] or []
            for b in branches:
                coord_raw = None
                if isinstance(b, dict):
                    coord_raw = b.get('พิกัด')
                elif isinstance(b, (list, tuple)) and len(b) > 2:
                    coord_raw = b[2]
                
                lat, lon = แปลงพิกัด(coord_raw)
                if lat is not None and lon is not None:
                    return True
        except Exception:
            pass
    
    # ตรวจสอบคอลัมน์ lat/lon
    lat_candidates = [c for c in row.index if c and 'lat' in c.lower()]
    lon_candidates = [c for c in row.index if c and ('lon' in c.lower() or 'lng' in c.lower())]
    
    if lat_candidates and lon_candidates:
        try:
            lat_val = row[lat_candidates[0]]
            lon_val = row[lon_candidates[0]]
            
            if isinstance(lat_val, (int, float)) and isinstance(lon_val, (int, float)):
                lat_f, lon_f = float(lat_val), float(lon_val)
            else:
                lat_f, lon_f = แปลงพิกัด([lat_val, lon_val])
            
            if lat_f is not None and lon_f is not None:
                return True
        except Exception:
            pass
    
    return False


@st.cache_data
def สร้างจุดหมุด(df_json):
    """สร้างจุดหมุดจาก DataFrame (มีแคชเพื่อประสิทธิภาพ)"""
    df = pd.read_json(io.StringIO(df_json), orient='split')
    markers = []
    
    # วิธีที่ 1: ดึงข้อมูลจากคอลัมน์ 'สาขา'
    if 'สาขา' in df.columns:
        for _, row in df.iterrows():
            company = row.get('ชื่อบริษัท', '')
            license_no = row.get('เลขที่ใบอนุญาต', '')
            branches = row.get('สาขา', [])
            
            if isinstance(branches, list):
                for b in branches:
                    addr = phone = coord_raw = None
                    
                    if isinstance(b, dict):
                        addr = b.get('ที่อยู่')
                        phone = b.get('โทรศัพท์')
                        coord_raw = b.get('พิกัด')
                    elif isinstance(b, (list, tuple)) and len(b) >= 3:
                        addr, phone, coord_raw = b[0], b[1], b[2]
                    
                    lat, lon = แปลงพิกัด(coord_raw)
                    # ตรวจสอบว่าพิกัดถูกต้องและไม่เป็น NaN
                    if (lat is not None and lon is not None and 
                        math.isfinite(lat) and math.isfinite(lon)):
                        popup = f"🏢 {company}\n📍 {addr or ''}\n📞 {phone or ''}\n📋 {license_no}"
                        markers.append({'lat': lat, 'lon': lon, 'popup': popup})
    
    # วิธีที่ 2: ใช้คอลัมน์ lat/lon
    if not markers:
        lat_cols = [c for c in df.columns if c and 'lat' in c.lower()]
        lon_cols = [c for c in df.columns if c and ('lon' in c.lower() or 'lng' in c.lower())]
        
        if lat_cols and lon_cols:
            lat_col, lon_col = lat_cols[0], lon_cols[0]
            
            for _, row in df.iterrows():
                try:
                    lat_raw = row.get(lat_col)
                    lon_raw = row.get(lon_col)
                    
                    if isinstance(lat_raw, (int, float)) and isinstance(lon_raw, (int, float)):
                        lat, lon = float(lat_raw), float(lon_raw)
                    else:
                        lat, lon = แปลงพิกัด([lat_raw, lon_raw])
                    
                    # ตรวจสอบว่าพิกัดถูกต้องและไม่เป็น NaN
                    if (lat is not None and lon is not None and 
                        math.isfinite(lat) and math.isfinite(lon)):
                        popup = f"🏢 {row.get('ชื่อบริษัท', '')}\n📋 {row.get('เลขที่ใบอนุญาต', '')}"
                        markers.append({'lat': lat, 'lon': lon, 'popup': popup})
                except Exception:
                    continue
    
    return markers


def สร้างแผนที่(markers):
    """สร้างแผนที่ folium พร้อมจุดหมุด"""
    # คำนวณจุดกึ่งกลาง
    center_lat, center_lon = 13.736, 100.518  # กรุงเทพฯ เป็นค่าเริ่มต้น
    if markers:
        # กรองเฉพาะพิกัดที่ถูกต้อง (ไม่มี None, NaN)
        valid_lats = [m['lat'] for m in markers if m.get('lat') is not None and math.isfinite(float(m['lat']))]
        valid_lons = [m['lon'] for m in markers if m.get('lon') is not None and math.isfinite(float(m['lon']))]
        
        if valid_lats and valid_lons:
            center_lat = sum(valid_lats) / len(valid_lats)
            center_lon = sum(valid_lons) / len(valid_lons)
    
    # ตรวจสอบว่าพิกัดกึ่งกลางถูกต้อง
    if not (math.isfinite(center_lat) and math.isfinite(center_lon)):
        center_lat, center_lon = 13.736, 100.518
    
    # สร้างแผนที่
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7 if len(markers) > 1 else 12,
        tiles='OpenStreetMap'
    )
    
    # เพิ่มจุดสำคัญ - หอไอเฟิล (จุดศูนย์กลางระบบ)
    folium.Marker(
        location=[13.7818, 100.5351],
        popup=folium.Popup("<b>🗼 จุดศูนย์กลางระบบ</b><br>หอไอเฟิล", max_width=250),
        tooltip="🗼 จุดศูนย์กลางระบบ",
        icon=folium.Icon(color='red', icon='tower-broadcast', prefix='fa')
    ).add_to(m)
    
    # เพิ่มจุดหมุดพร้อมการจัดกลุ่ม
    mc = MarkerCluster().add_to(m)
    
    for mrec in markers:
        lat, lon = mrec.get('lat'), mrec.get('lon')
        
        if lat is None or lon is None:
            continue
        
        try:
            lat_f, lon_f = float(lat), float(lon)
            if not (math.isfinite(lat_f) and math.isfinite(lon_f)):
                continue
            
            popup_text = mrec.get('popup', '')
            tooltip_text = popup_text.split('\n')[0] if popup_text else None
            
            folium.Marker(
                location=[lat_f, lon_f],
                popup=folium.Popup(popup_text, max_width=300),
                tooltip=tooltip_text,
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(mc)
        except Exception:
            continue
    
    return m


# ==================== แอปพลิเคชันหลัก ====================

def main():
    st.title("📍 แผนที่บริษัทผลิตภัณฑ์")
    st.markdown("---")
    
    # โหลดข้อมูล
    try:
        df = โหลดข้อมูล('data/merged_pico_companies.json')
    except FileNotFoundError:
        st.error("❌ ไม่พบไฟล์ data/merged_pico_companies.json")
        st.info("💡 กรุณาตรวจสอบว่าไฟล์อยู่ในโฟลเดอร์ที่ถูกต้อง")
        return
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการโหลดข้อมูล: {e}")
        return
    
    # แถบด้านข้างสำหรับตัวกรอง
    st.sidebar.header("🔍 ตัวกรองข้อมูล")
    st.sidebar.markdown("---")
    
    # ตัวกรองภูมิภาค
    regions = ['ทั้งหมด']
    if 'ภูมิภาค' in df.columns:
        regions += sorted(df['ภูมิภาค'].dropna().unique().tolist())
    selected_region = st.sidebar.selectbox('🗺️ ภูมิภาค', regions, index=0)
    
    # ตัวกรองจังหวัด
    prov_candidates = []
    if 'จังหวัด' in df.columns:
        if selected_region != 'ทั้งหมด' and 'ภูมิภาค' in df.columns:
            prov_candidates = df.loc[df['ภูมิภาค'] == selected_region, 'จังหวัด'].dropna().unique().tolist()
        else:
            prov_candidates = df['จังหวัด'].dropna().unique().tolist()
    provinces = ['ทั้งหมด'] + sorted(prov_candidates)
    selected_province = st.sidebar.selectbox('🏙️ จังหวัด', provinces, index=0)
    
    # ตัวกรองชื่อบริษัท
    name_query = st.sidebar.text_input('🏢 ค้นหาชื่อบริษัท', placeholder='พิมพ์ชื่อบริษัท...')
    
    # ตัวกรองพิกัด
    coords_only = st.sidebar.checkbox('📌 แสดงเฉพาะที่มีพิกัด', value=False)
    
    st.sidebar.markdown("---")
    
    # ใช้ตัวกรอง
    filtered_df = df.copy()
    
    if selected_region != 'ทั้งหมด':
        filtered_df = filtered_df[filtered_df['ภูมิภาค'] == selected_region]
    
    if selected_province != 'ทั้งหมด':
        filtered_df = filtered_df[filtered_df['จังหวัด'] == selected_province]
    
    if name_query and 'ชื่อบริษัท' in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df['ชื่อบริษัท'].astype(str).str.contains(name_query, case=False, na=False)
        ]
    
    if coords_only:
        filtered_df = filtered_df[filtered_df.apply(แถวมีพิกัด, axis=1)]
    
    filtered_df = filtered_df.reset_index(drop=True)
    
    # แสดงสถิติ
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 จำนวนบริษัททั้งหมด", f"{len(df):,}")
    with col2:
        st.metric("✅ บริษัทที่กรอง", f"{len(filtered_df):,}")
    
    # สร้างจุดหมุด (มีแคช)
    df_json = filtered_df.to_json(orient='split')
    markers = สร้างจุดหมุด(df_json)
    
    with col3:
        st.metric("📍 จุดบนแผนที่", f"{len(markers):,}")
    
    st.markdown("---")
    
    # สร้างและแสดงแผนที่
    if markers:
        m = สร้างแผนที่(markers)
        st_folium(m, width=1200, height=700, returned_objects=[])
    else:
        st.warning("⚠️ ไม่พบข้อมูลตำแหน่งที่ตรงกับการกรอง")
        st.info("💡 ลองปรับเปลี่ยนตัวกรองเพื่อดูข้อมูลเพิ่มเติม")
        
    # แสดงตารางข้อมูล
    if st.sidebar.checkbox("📊 แสดงตารางข้อมูล", value=False):
        st.markdown("---")
        st.subheader("📋 ตารางข้อมูลที่กรอง")
        
        # เลือกคอลัมน์ที่จะแสดง
        display_cols = [col for col in ['ชื่อบริษัท', 'เลขที่ใบอนุญาต', 'จังหวัด', 'ภูมิภาค'] 
                       if col in filtered_df.columns]
        
        if display_cols:
            st.dataframe(
                filtered_df[display_cols], 
                use_container_width=True,
                height=400
            )
        else:
            st.dataframe(filtered_df, use_container_width=True, height=400)
        
        # ปุ่มดาวน์โหลด
        csv = filtered_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 ดาวน์โหลดข้อมูล CSV",
            data=csv,
            file_name="filtered_companies.csv",
            mime="text/csv"
        )
    
    # ข้อมูลเพิ่มเติมในแถบด้านข้าง
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **💡 วิธีใช้งาน:**
    - เลือกภูมิภาคและจังหวัดเพื่อกรองข้อมูล
    - ค้นหาชื่อบริษัทที่ต้องการ
    - คลิกที่จุดบนแผนที่เพื่อดูรายละเอียด
    - จุดหมุดจะจัดกลุ่มอัตโนมัติเมื่อซูมออก
    """)


if __name__ == "__main__":
    main()