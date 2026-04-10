import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import os
import warnings
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from networkx.algorithms.community import greedy_modularity_communities

warnings.filterwarnings('ignore')

# ==========================================
# 1. CẤU HÌNH TRANG WEB
# ==========================================
st.set_page_config(page_title="Financial Network Analysis", layout="wide")
st.title("📊 Hệ Thống Phân Tích Mạng Lưới Tài Chính (Partial Correlation)")
st.markdown("> **Ghi chú:** Hệ thống đang sử dụng **Partial Correlation** để loại bỏ nhiễu thị trường, giúp soi rõ cấu trúc ngành thực sực.")

# ==========================================
# 2. TẢI DỮ LIỆU
# ==========================================
@st.cache_data
def load_data():
    file_path = 'log_returns.csv'
    if os.path.exists(file_path):
        return pd.read_csv(file_path, index_col=0, parse_dates=True)
    return None

log_returns = load_data()

if log_returns is None:
    st.error("❌ Không tìm thấy file `log_returns.csv`.")
    st.stop()

data_start = log_returns.index.min().date()
data_end = log_returns.index.max().date()

# ==========================================
# 3. SIDEBAR
# ==========================================
st.sidebar.header("⚙️ 1. Cấu Hình Thời Gian")
time_mode = st.sidebar.radio("Phương thức:", ["Chọn Start/End Date", "Dùng Rolling Window"])

if time_mode == "Chọn Start/End Date":
    start_dt = st.sidebar.date_input("Ngày bắt đầu:", value=data_start, min_value=data_start, max_value=data_end)
    end_dt = st.sidebar.date_input("Ngày kết thúc:", value=data_end, min_value=data_start, max_value=data_end)
else:
    end_dt = st.sidebar.date_input("Ngày kết thúc:", value=data_end, min_value=data_start, max_value=data_end)
    window_size = st.sidebar.slider("Window (phiên):", 22, 252, 126)
    idx = log_returns.index.get_indexer([pd.to_datetime(end_dt)], method='pad')[0]
    start_dt = log_returns.index[max(0, idx - window_size + 1)].date()
    st.sidebar.info(f"Ngày bắt đầu: {start_dt}")

graph_type = st.sidebar.radio("Loại mạng lưới:", ["MST (Cấu trúc tối giản)", "PMFG (Cấu trúc phẳng & Phân cụm)"])

st.sidebar.header("🎨 2. Trực Quan Hóa")
centrality_metric = st.sidebar.selectbox("Định cỡ Node theo:", ["Betweenness", "Degree", "Closeness"])
layout_mode = st.sidebar.radio("Chế độ hiển thị:", ["Động (Vật lý)", "Tĩnh (Kamada-Kawai)"])
run_btn = st.sidebar.button("🚀 Chạy Phân Tích", type="primary")

# ==========================================
# 4. HÀM TÍNH TOÁN (PARTIAL CORRELATION)
# ==========================================

def compute_partial_correlation(df):
    """
    Tính ma trận Partial Correlation từ ma trận nghịch đảo (Precision Matrix).
    """
    # 1. Tính ma trận Pearson gốc
    C = df.corr().values
    
    # 2. Tính ma trận nghịch đảo (Pseudo-inverse để tránh lỗi ma trận suy biến)
    try:
        P = np.linalg.pinv(C)
    except np.linalg.LinAlgError:
        return df.corr() # Fallback nếu lỗi nặng

    # 3. Áp dụng công thức rho_ij = -P_ij / sqrt(P_ii * P_jj)
    d = np.diag(1 / np.sqrt(np.diag(P)))
    partial_corr_values = -d @ P @ d
    np.fill_diagonal(partial_corr_values, 1.0)
    
    return pd.DataFrame(partial_corr_values, index=df.columns, columns=df.columns)

@st.cache_data
def get_matrices(start_date, end_date):
    window_data = log_returns.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
    if len(window_data) < 2: return None, None, 0
    
    # THAY ĐỔI CHÍNH: Tính Partial Corr thay vì Pearson Corr
    partial_corr = compute_partial_correlation(window_data)
    
    # Tính khoảng cách dựa trên Partial Corr
    # d = sqrt(2 * (1 - rho_partial))
    dist_matrix = np.sqrt(2 * (1 - partial_corr).clip(lower=0))
    return partial_corr, dist_matrix, len(window_data)

@st.cache_resource
def compute_algorithms(corr_matrix, dist_matrix, g_type):
    G_full = nx.Graph()
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            # Lưu trọng số là khoảng cách
            G_full.add_edge(cols[i], cols[j], weight=dist_matrix.iloc[i, j])

    if "MST" in g_type:
        target_graph = nx.minimum_spanning_tree(G_full, weight='weight')
        node_to_cluster = None
    else:
        # Xây dựng PMFG
        sorted_edges = sorted(G_full.edges(data=True), key=lambda x: x[2]['weight'])
        target_graph = nx.Graph()
        target_graph.add_nodes_from(G_full.nodes())
        max_edges = 3 * (len(cols) - 2)
        for u, v, data in sorted_edges:
            if target_graph.number_of_edges() >= max_edges: break
            target_graph.add_edge(u, v, **data)
            if not nx.check_planarity(target_graph)[0]:
                target_graph.remove_edge(u, v)
        
        communities = greedy_modularity_communities(target_graph, weight='weight')
        node_to_cluster = {node: cid + 1 for cid, comm in enumerate(communities) for node in comm}
        
    return target_graph, node_to_cluster

# ... (Giữ nguyên các hàm calculate_centrality, setup_pyvis_network) ...

def calculate_centrality(graph, metric_name):
    if "Betweenness" in metric_name:
        return nx.betweenness_centrality(graph, weight='weight')
    elif "Degree" in metric_name:
        return nx.degree_centrality(graph)
    else:
        return nx.closeness_centrality(graph, distance='weight')

def setup_pyvis_network():
    return Network(height='700px', width='100%', bgcolor='#ffffff', font_color='black', select_menu=True, cdn_resources='remote')

# ==========================================
# 5. XỬ LÝ GIAO DIỆN & VẼ ĐỒ THỊ
# ==========================================
if run_btn and start_dt < end_dt:
    corr_matrix, dist_matrix, num_days = get_matrices(start_dt, end_dt)
    if corr_matrix is None:
        st.error("Không đủ dữ liệu.")
        st.stop()

    with st.spinner('Đang tính toán Partial Correlation và xây dựng mạng lưới...'):
        target_graph, node_to_cluster = compute_algorithms(corr_matrix, dist_matrix, graph_type)
        cent_dict = calculate_centrality(target_graph, centrality_metric)
        max_cent = max(cent_dict.values()) if cent_dict else 1

        # Chú thích
        st.markdown("### 📖 Chú thích Biểu đồ (Partial Correlation)")
        st.info("Mạng lưới này đã loại bỏ sự tương quan ảo do thị trường chung gây ra.")
        
        # Thiết lập màu sắc
        net = setup_pyvis_network()
        cmap_pmfg = cm.get_cmap('tab20', 20)
        cmap_mst = cm.get_cmap('YlOrRd')

        # Xử lý Layout
        if layout_mode == "Tĩnh (Kamada-Kawai)":
            pos = nx.kamada_kawai_layout(target_graph, weight='weight')
            scale = 800
            net.toggle_physics(False)
        else:
            pos, scale = None, 1

        # Vẽ Nodes
        for node in target_graph.nodes():
            size = (cent_dict[node] / max_cent) * 40 + 15
            if node_to_cluster:
                hex_color = mcolors.rgb2hex(cmap_pmfg(node_to_cluster[node] % 20))
                title = f"Mã: {node}\nCụm: {node_to_cluster[node]}"
            else:
                hex_color = mcolors.rgb2hex(cmap_mst(0.3 + 0.7 * (cent_dict[node] / max_cent)))
                title = f"Mã: {node}\nCentrality: {cent_dict[node]:.4f}"
            
            if pos:
                net.add_node(node, label=str(node), size=size, title=title, color=hex_color,
                             x=float(pos[node][0]*scale), y=float(pos[node][1]*scale))
            else:
                net.add_node(node, label=str(node), size=size, title=title, color=hex_color)

        # Vẽ Edges
        for u, v, data in target_graph.edges(data=True):
            w = data['weight']
            net.add_edge(u, v, value=max(0.3, 3-w), title=f"Dist: {w:.3f}", color="#cccccc")

        net.save_graph("temp.html")
        components.html(open("temp.html", 'r').read(), height=720)

        # Top 5
        st.markdown("### 🏆 Top 5 Mã trung tâm")
        top_5 = sorted(cent_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        c1, c2, c3, c4, c5 = st.columns(5)
        for i, (m, v) in enumerate(top_5): [c1, c2, c3, c4, c5][i].metric(m, f"{v:.4f}")