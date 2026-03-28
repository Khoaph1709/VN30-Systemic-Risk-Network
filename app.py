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
st.set_page_config(page_title="Network Analysis", layout="wide")
st.title("📊 Hệ Thống Phân Tích Mạng Lưới Tài Chính Động & Phân Cụm Ngành")

# ==========================================
# 2. TẢI DỮ LIỆU CƠ BẢN (Đã tối ưu Cache)
# ==========================================
@st.cache_data
def load_data():
    file_path = 'log_returns.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return df
    return None

log_returns = load_data()

if log_returns is None:
    st.error("❌ Không tìm thấy file `log_returns.csv`. Vui lòng để file này cùng thư mục với `app.py`.")
    st.stop()

data_start = log_returns.index.min().date()
data_end = log_returns.index.max().date()

# ==========================================
# 3. THIẾT KẾ SIDEBAR
# ==========================================
st.sidebar.header("⚙️ 1. Bảng Điều Khiển Thời Gian")

time_mode = st.sidebar.radio("Phương thức chọn thời gian:", ["Chọn Start/End Date", "Dùng Rolling Window"])

if time_mode == "Chọn Start/End Date":
    start_dt = st.sidebar.date_input("Ngày bắt đầu:", value=data_start, min_value=data_start, max_value=data_end)
    end_dt = st.sidebar.date_input("Ngày kết thúc:", value=data_end, min_value=data_start, max_value=data_end)
else:
    end_dt = st.sidebar.date_input("Ngày kết thúc:", value=data_end, min_value=data_start, max_value=data_end)
    window_size = st.sidebar.slider("Window (số phiên):", min_value=22, max_value=252, value=158, step=1)
    
    end_dt_pd = pd.to_datetime(end_dt)
    idx = log_returns.index.get_indexer([end_dt_pd], method='pad')[0]
    start_idx = max(0, idx - window_size + 1)
    start_dt = log_returns.index[start_idx].date()
    st.sidebar.info(f"Ngày bắt đầu tương ứng: {start_dt}")

graph_type = st.sidebar.radio("Chọn loại phân tích:", ["MST (Minimum Spanning Tree)", "PMFG (Planar) & DBHT Clustering"])

st.sidebar.markdown("---")
st.sidebar.header("🎨 2. Tùy Chỉnh Trực Quan")
centrality_metric = st.sidebar.selectbox(
    "Định cỡ Nốt (Node Size) theo:", 
    ["Betweenness (Quyền lực cầu nối)", "Degree (Số lượng liên kết)", "Closeness (Độ nhạy thông tin)"]
)

run_btn = st.sidebar.button("🚀 Chạy Phân Tích Đồ Thị", type="primary")

if start_dt >= end_dt:
    st.sidebar.error("Lỗi: Ngày bắt đầu phải diễn ra trước Ngày kết thúc!")

# ==========================================
# 4. CÁC HÀM CỐT LÕI (ĐÃ TỐI ƯU HÓA CACHE)
# ==========================================

# 4.1. Caching Ma trận để không phải tính lại Pearson mỗi lần bấm nút
@st.cache_data
def get_matrices(start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    window_data = log_returns.loc[start_date:end_date]
    if window_data.empty or len(window_data) < 2:
        return None, None, 0
    
    corr_matrix = window_data.corr(method='pearson')
    dist_matrix = np.sqrt(2 * (1 - corr_matrix).clip(lower=0))
    return corr_matrix, dist_matrix, len(window_data)

# 4.2. Caching PMFG (Vì thuật toán check Planarity rất nặng)
@st.cache_resource
def compute_algorithms(corr_matrix, dist_matrix, g_type):
    # Khởi tạo đồ thị đầy đủ
    G_full = nx.Graph()
    cols = corr_matrix.columns
    G_full.add_nodes_from(cols)
    
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            G_full.add_edge(cols[i], cols[j], weight=dist_matrix.iloc[i, j])

    # Chạy thuật toán tương ứng
    if "MST" in g_type:
        target_graph = nx.minimum_spanning_tree(G_full, weight='weight')
        node_to_cluster = None
    else:
        # Xây dựng PMFG
        sorted_edges = sorted(G_full.edges(data=True), key=lambda x: x[2]['weight'])
        N = G_full.number_of_nodes()
        target_graph = nx.Graph()
        target_graph.add_nodes_from(G_full.nodes())
        max_edges = 3 * (N - 2)
        
        for u, v, data in sorted_edges:
            if target_graph.number_of_edges() >= max_edges: break
            target_graph.add_edge(u, v, **data)
            if not nx.check_planarity(target_graph)[0]:
                target_graph.remove_edge(u, v)
        
        # Chạy DBHT Clustering
        communities = greedy_modularity_communities(target_graph, weight='weight')
        node_to_cluster = {node: cid + 1 for cid, comm in enumerate(communities) for node in comm}
        
    return target_graph, node_to_cluster

# 4.3. Tính toán Node Size (Rất nhẹ, không cần cache)
def calculate_centrality(graph, metric_name):
    if "Betweenness" in metric_name:
        return nx.betweenness_centrality(graph, weight='weight')
    elif "Degree" in metric_name:
        return nx.degree_centrality(graph)
    else:
        return nx.closeness_centrality(graph, distance='weight')

# 4.4. Cấu hình Pyvis (ĐÃ SỬA LỖI TRẮNG MÀN HÌNH)
def setup_pyvis_network():
    # Thêm cdn_resources='remote' là CHÌA KHÓA để select_menu hoạt động trên Streamlit
    net = Network(height='700px', width='100%', bgcolor='#ffffff', font_color='black', 
                  select_menu=True, cdn_resources='remote')
    net.force_atlas_2based(
        gravity=-50, central_gravity=0.01, spring_length=150, spring_strength=0.08, damping=0.4, overlap=0
    )
    return net

# ==========================================
# 5. XỬ LÝ SỰ KIỆN GIAO DIỆN
# ==========================================
if run_btn and start_dt < end_dt:
    corr_matrix, dist_matrix, num_days = get_matrices(start_dt, end_dt)
    if corr_matrix is None:
        st.error("Không đủ dữ liệu trong khoảng thời gian này.")
        st.stop()
        
    st.info(f"Đang phân tích **{len(corr_matrix.columns)}** mã chứng khoán trong **{num_days}** phiên giao dịch...")
    
    metric_short_name = centrality_metric.split(" ")[0]

    with st.spinner('Đang xử lý mạng lưới...'):
        # Lấy đồ thị từ bộ nhớ đệm (Nhanh gấp 10 lần)
        target_graph, node_to_cluster = compute_algorithms(corr_matrix, dist_matrix, graph_type)
        
        # Tính toán Centrality động
        cent_dict = calculate_centrality(target_graph, centrality_metric)
        max_cent = max(cent_dict.values()) if cent_dict else 1
        
        # Vẽ đồ thị
        net = setup_pyvis_network()
        cmap = cm.get_cmap('Set3', 12) 
        
        for node in target_graph.nodes():
            # MST kích thước nốt to hơn PMFG một chút
            base_size = 30 if "MST" in graph_type else 20
            base_min = 15 if "MST" in graph_type else 10
            
            size = (cent_dict[node] / max_cent) * base_size + base_min if max_cent > 0 else base_min
            
            if node_to_cluster: # Dành cho PMFG
                cluster_id = node_to_cluster.get(node, 0)
                hex_color = mcolors.rgb2hex(cmap(cluster_id % 12))
                hover_text = f"Mã: {node}\nCụm ngành: {cluster_id}\n{metric_short_name}: {cent_dict[node]:.4f}"
            else: # Dành cho MST
                intensity = int(255 - (cent_dict[node] / max_cent) * 150) if max_cent > 0 else 255
                hex_color = f"#ff{intensity:02x}{intensity:02x}"
                hover_text = f"Mã: {node}\n{metric_short_name}: {cent_dict[node]:.4f}"
                
            net.add_node(node, label=str(node), size=size, title=hover_text, color=hex_color)
        
        for source, target, data in target_graph.edges(data=True):
            weight = data['weight']
            # Cạnh MST dày hơn PMFG
            edge_width = max(0.5, 3 - weight) if "MST" in graph_type else max(0.3, 2 - weight)
            net.add_edge(source, target, value=edge_width, title=f"Distance: {weight:.3f}", color="#cccccc")
        
        # Lưu và hiển thị
        html_file = "interactive_graph.html"
        net.save_graph(html_file)
        with open(html_file, 'r', encoding='utf-8') as f:
            components.html(f.read(), height=720, scrolling=False)
            
        # --- HIỂN THỊ KẾT QUẢ VĂN BẢN ---
        st.markdown(f"### 🏆 Top 5 Mã Dẫn Đầu ({metric_short_name})")
        top_5 = sorted(cent_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        cols = st.columns(5)
        for i, (node, val) in enumerate(top_5): cols[i].metric(label=f"Top {i+1}: {node}", value=f"{val:.4f}")    
            
        if node_to_cluster:
            st.markdown("### 🏢 Kết quả Phân Cụm Ngành DBHT")
            clusters = {}
            for node, cluster_id in node_to_cluster.items():
                if cluster_id not in clusters: clusters[cluster_id] = []
                clusters[cluster_id].append(node)
            
            sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
            for cluster_id, nodes in sorted_clusters:
                with st.expander(f"Cụm {cluster_id} ({len(nodes)} mã)"):
                    st.write(", ".join(nodes))