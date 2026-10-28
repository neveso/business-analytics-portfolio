import streamlit as st
import pandas as pd
import numpy as np
import boto3
import json
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64
from io import BytesIO

# Configuração da página
st.set_page_config(
    page_title="OLX - Avaliador de Smartphones",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para melhorar o visual
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #CBF8F5 0%, #24E7DA 100%);
        padding: 2rem 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: #012C2D;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .price-card {
        background: linear-gradient(135deg, #CBF8F5 0%, #24E7DA 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: #012C2D;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #24E7DA;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        color: #333;
    }
    
    .metric-card {
        background: #24E7DA;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 0.5rem;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #CBF8F5 0%, #24E7DA 100%);
        color: #012C2D;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    
    .sidebar .stSelectbox > div > div {
        background-color: #012C2D;
    }
    
    .price-low { color: #28a745; font-weight: bold; }
    .price-medium { color: #ffc107; font-weight: bold; }
    .price-high { color: #fd7e14; font-weight: bold; }
    .price-very-high { color: #dc3545; font-weight: bold; }
    
    .help-tooltip {
        background-color: #012C2D;
        border: 1px solid #012C2D;
        border-radius: 5px;
        padding: 0.5rem;
        margin-top: 0.25rem;
        font-size: 0.9rem;
        color: #495057;
    }
   
</style>
""", unsafe_allow_html=True)

# Configuração AWS 
AWS_REGION = 'us-east-1' 
ENDPOINT_NAME = 'price-range-endpoint' 


@st.cache_resource
def get_predictor():
    try:
        sagemaker_client = boto3.client('sagemaker', region_name=AWS_REGION)
        status = sagemaker_client.describe_endpoint(EndpointName=ENDPOINT_NAME)['EndpointStatus']
        if status != 'InService':
            st.error("❌ Endpoint SageMaker não está ativo")
            return None
        predictor = Predictor(
            endpoint_name=ENDPOINT_NAME,
            serializer=CSVSerializer(),
            deserializer=CSVDeserializer()
        )
        return predictor
    except Exception as e:
        st.error(f"❌ Erro ao conectar com o modelo: {e}")
        return None

# Função de previsão 
def predict_price_range(features_dict):
    predictor = get_predictor()
    if predictor is None:
        return None
    # DataFrame original
    df = pd.DataFrame([features_dict])
    # Feature engineering
    df['screen_area'] = df['sc_h'] * df['sc_w']
    df['pixel_density'] = np.sqrt(df['px_height']**2 + df['px_width']**2) / (df['screen_area'] + 1e-6)
    df['ram_per_core'] = df['ram'] / (df['n_cores'] + 1e-6)
    df['camera_total'] = df['pc'] + df['fc']
    df['weight_to_battery_ratio'] = df['mobile_wt'] / (df['battery_power'] + 1e-6)
    df['has_connectivity'] = df[['blue','wifi','three_g','four_g']].sum(axis=1)
    df['advanced_features'] = df[['touch_screen','dual_sim']].sum(axis=1)
    # Ordenar colunas
    cols = [
        'battery_power','blue','clock_speed','dual_sim','fc','four_g','int_memory',
        'm_dep','mobile_wt','n_cores','pc','px_height','px_width','ram','sc_h','sc_w',
        'talk_time','three_g','touch_screen','wifi','screen_area','pixel_density',
        'ram_per_core','camera_total','weight_to_battery_ratio','has_connectivity','advanced_features'
    ]
    df = df[cols]
    # Chamada ao endpoint
    csv_data = df.to_csv(header=False, index=False).strip()
    result = predictor.predict(csv_data)
    probs = [float(p) for p in result[0]]
    prediction = max(range(len(probs)), key=lambda i: probs[i])
    return prediction


# Interface

# Função para mostrar ajuda
def show_help(text):
    with st.popover("❓"):
        st.markdown(text)

# Cabeçalho principal
st.markdown("""
<div class="main-header">
    <h1>📱 OLX - Avaliador de Smartphones</h1>
    <p style="font-size: 1.2rem; margin: 0;">Descubra a gama de preço ideal para o seu smartphone</p>
</div>
""", unsafe_allow_html=True)

# Layout principal
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🔧 Características do Seu Smartphone")
    
    # Separar em categorias para melhor UX
    tab1, tab2, tab3, tab4 = st.tabs(["📱 Básico", "⚡ Performance", "📸 Câmara", "🔋 Bateria & Conectividade"])
    
    # Inicializar dicionário de características
    features = {}
    
    with tab1:
        col1_1, col1_2 = st.columns(2)
        
        with col1_1:
            features['mobile_wt'] = st.slider(
                "Peso (gramas)", 
                min_value=80, max_value=200, value=150,
                help="💡 **Como encontrar**: Especificações do fabricante ou pesquisar online o modelo do seu telefone. Telefones mais leves são geralmente mais caros."
            )
            
            features['m_dep'] = st.slider(
                "Espessura (cm)", 
                min_value=0.1, max_value=1.0, value=0.5, step=0.05,
                help="💡 **Como medir**: Use uma régua para medir a espessura do telefone. Telefones mais finos são geralmente mais premium."
            )
            
            features['sc_h'] = st.slider(
                "Altura da Tela (cm)", 
                min_value=5.0, max_value=19.0, value=12.0, step=0.5,
                help="💡 **Como medir**: Meça apenas a área da tela (sem as bordas) com uma régua, de cima para baixo. Telas maiores aumentam o valor."
            )
        
        with col1_2:
            features['sc_w'] = st.slider(
                "Largura da Tela (cm)", 
                min_value=0.0, max_value=18.0, value=5.0, step=0.5,
                help="💡 **Como medir**: Meça apenas a área da tela (sem as bordas) com uma régua, da esquerda para a direita."
            )
            
            features['px_height'] = st.number_input(
                "Resolução - Altura (pixels)", 
                min_value=0, max_value=1960, value=1080,
                help="💡 **Como encontrar**: Vá a Definições > Ecrã > Resolução, ou pesquise o modelo online. Ex: 1920x1080 significa altura=1920."
            )
            
            features['px_width'] = st.number_input(
                "Resolução - Largura (pixels)", 
                min_value=500, max_value=1998, value=1080,
                help="💡 **Como encontrar**: Vá a Definições > Ecrã > Resolução, ou pesquise o modelo online. Ex: 1920x1080 significa largura=1080."
            )
    
    with tab2:
        col2_1, col2_2 = st.columns(2)
        
        with col2_1:
            features['ram'] = st.selectbox(
                "Memória RAM (MB)", 
                options=[256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 3998],
                index=4,
                help="💡 **Como encontrar**: Android: Definições > Sobre o telefone > RAM. iPhone: pesquise o modelo online. Mais RAM = melhor performance e maior valor."
            )
            
            features['int_memory'] = st.selectbox(
                "Armazenamento Interno (GB)", 
                options=[2, 4, 8, 16, 32, 64],
                index=4,
                help="💡 **Como encontrar**: Definições > Armazenamento. Mostra espaço total disponível. Mais armazenamento = maior valor."
            )
        
        with col2_2:
            features['clock_speed'] = st.slider(
                "Velocidade do Processador (GHz)", 
                min_value=0.5, max_value=3.0, value=1.5, step=0.1,
                help="💡 **Como encontrar**: Apps como CPU-Z ou AnTuTu, ou pesquise o modelo online. Velocidade mais alta = melhor performance."
            )
            
            features['n_cores'] = st.selectbox(
                "Número de Núcleos do Processador", 
                options=[1, 2, 3, 4, 5, 6, 7, 8],
                index=3,
                help="💡 **Como encontrar**: Apps como CPU-Z, ou pesquise as especificações do modelo. Mais núcleos = melhor multitasking."
            )
    
    with tab3:
        col3_1, col3_2 = st.columns(2)
        
        with col3_1:
            features['pc'] = st.slider(
                "Câmara Principal (MP)", 
                min_value=0, max_value=20, value=8,
                help="💡 **Como encontrar**: App da câmara > definições, ou especificações do modelo. Mais megapixels = fotos de maior qualidade."
            )
        
        with col3_2:
            features['fc'] = st.slider(
                "Câmara Frontal (MP)", 
                min_value=0, max_value=19, value=2,
                help="💡 **Como encontrar**: App da câmara > trocar para frontal > definições. Importante para selfies e videochamadas."
            )
    
    with tab4:
        col4_1, col4_2 = st.columns(2)
        
        with col4_1:
            features['battery_power'] = st.slider(
                "Capacidade da Bateria (mAh)", 
                min_value=501, max_value=1998, value=1200,
                help="💡 **Como encontrar**: Definições > Bateria > Informações da bateria. Ou remover a bateria (se possível) para ver a etiqueta. Maior capacidade = mais autonomia."
            )
            
            features['talk_time'] = st.slider(
                "Autonomia em Chamada (horas)", 
                min_value=2, max_value=20, value=10,
                help="💡 **Como encontrar**: Especificações do fabricante ou testes online do modelo. Indica quantas horas dura a bateria numa chamada contínua."
            )
        
        with col4_2:
            st.markdown("**Conectividade & Funcionalidades**")
            st.markdown("*💡 Vá a Definições > Conectividade para verificar*")
            
            features['blue'] = 1 if st.checkbox("Bluetooth", value=True, help="💡 Definições > Bluetooth. Praticamente todos os smartphones têm.") else 0
            features['wifi'] = 1 if st.checkbox("Wi-Fi", value=True, help="💡 Definições > Wi-Fi. Praticamente todos os smartphones têm.") else 0
            features['three_g'] = 1 if st.checkbox("3G", value=True, help="💡 Definições > Redes móveis. Tecnologia mais antiga mas ainda presente na maioria.") else 0
            features['four_g'] = 1 if st.checkbox("4G/LTE", value=True, help="💡 Definições > Redes móveis. Veja se aparece '4G' ou 'LTE' no topo do ecrã.") else 0
            features['dual_sim'] = 1 if st.checkbox("Dual SIM", value=False, help="💡 Verifique se tem 2 slots para cartão SIM. Permite usar 2 números no mesmo telefone.") else 0
            features['touch_screen'] = 1 if st.checkbox("Ecrã Táctil", value=True, help="💡 Se consegue tocar no ecrã para navegar, tem ecrã táctil. Praticamente todos os smartphones têm.") else 0

with col2:
    st.markdown("### 💰 Avaliação do Preço")
    
    if st.button("🎯 Avaliar Smartphone", use_container_width=True):
        with st.spinner("A analisar o seu smartphone..."):
            ## Debug: mostrar os valores enviados
            #st.markdown("#### 🔍 Debug - Valores Enviados:")
            #debug_df = pd.DataFrame([features])
            #debug_df['pixel_density'] = (
            #    debug_df['px_height'] * debug_df['px_width']
            #) / ((debug_df['sc_h'] * debug_df['sc_w']) + 1e-6)
            
            #st.dataframe(debug_df)
            
            prediction = predict_price_range(features)
            
            if prediction is not None:
                # Mapear previsão para descrição
                price_ranges = {
                    0: {"name": "Baixo Custo", "desc": "Smartphone económico", "color": "price-low", "emoji": "💚"},
                    1: {"name": "Custo Médio", "desc": "Boa relação qualidade-preço", "color": "price-medium", "emoji": "💛"},
                    2: {"name": "Custo Elevado", "desc": "Smartphone premium", "color": "price-high", "emoji": "🧡"},
                    3: {"name": "Custo Muito Elevado", "desc": "Topo de gama", "color": "price-very-high", "emoji": "❤️"}
                }
                
                result = price_ranges[prediction]
                
                # Exibir resultado
                st.markdown(f"""
                <div class="price-card">
                    <h2>{result['emoji']} {result['name']}</h2>
                    <p style="font-size: 1.2rem; margin: 0;">{result['desc']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Dicas personalizadas
                if prediction == 0:
                    st.info("💡 **Dica**: Este smartphone é ideal para utilizadores que procuram funcionalidades básicas a um preço acessível.")
                elif prediction == 1:
                    st.info("💡 **Dica**: Excelente escolha! Este smartphone oferece um bom equilíbrio entre características e preço.")
                elif prediction == 2:
                    st.info("💡 **Dica**: Smartphone premium com características avançadas. Ideal para utilizadores exigentes.")
                else:
                    st.info("💡 **Dica**: Topo de gama! Este smartphone possui as melhores características do mercado.")
    
    # Mostrar resumo das características
    st.markdown("### 📊 Resumo das Características")
    
    # Criar métricas visuais
    metrics_col1, metrics_col2 = st.columns(2)
    
    with metrics_col1:
        if 'ram' in features:
            st.metric("RAM", f"{features.get('ram', 0)} MB")
        if 'int_memory' in features:
            st.metric("Armazenamento", f"{features.get('int_memory', 0)} GB")
    
    with metrics_col2:
        if 'battery_power' in features:
            st.metric("Bateria", f"{features.get('battery_power', 0)} mAh")
        if 'pc' in features:
            st.metric("Câmara Principal", f"{features.get('pc', 0)} MP")

# Seção informativa
st.markdown("---")
st.markdown("### ℹ️ Como Funciona?")

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.markdown("""
    <div class="feature-card">
        <h4>🤖 Inteligência Artificial</h4>
        <p>Utilizamos algoritmos avançados de ML para analisar as características do seu smartphone.</p>
    </div>
    """, unsafe_allow_html=True)

with info_col2:
    st.markdown("""
    <div class="feature-card">
        <h4>📈 Análise de Mercado</h4>
        <p>O modelo foi treinado com dados reais do mercado para fornecer avaliações precisas.</p>
    </div>
    """, unsafe_allow_html=True)

with info_col3:
    st.markdown("""
    <div class="feature-card">
        <h4>⚡ Resultados Instantâneos</h4>
        <p>Obtenha a classificação de preço do seu smartphone em segundos!</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🏢 <strong>OLX Portugal</strong> - Ferramenta de Avaliação de Smartphones</p>
    <p>Desenvolvido pela SmartPrice com ❤️ para ajudar os nossos utilizadores a vender melhor.</p>
    <p><em>Projeto desenvolvido por Ana Rita Maganinho, Bárbara Pedrosa e Liliana Gestosa no âmbito da unidade curricular de Cloud & Cognitive Business Analytics (CCBA).</em></p>
</div>
""", unsafe_allow_html=True)

# Sidebar com informações adicionais
with st.sidebar:
    st.markdown("### 📱 Sobre esta Ferramenta")
    st.markdown("""
    Esta ferramenta utiliza **Inteligência Artificial** para classificar smartphones em diferentes gamas de preço:
    
    - **💚 Baixo Custo**: Smartphones económicos
    - **💛 Custo Médio**: Boa relação qualidade-preço  
    - **🧡 Custo Elevado**: Smartphones premium
    - **❤️ Custo Muito Elevado**: Topo de gama
    
    ### 🎯 Benefícios
    - ✅ Avaliação instantânea
    - ✅ Baseado em dados reais
    - ✅ Ajuda a definir preços competitivos
    - ✅ Aumenta as hipóteses de venda
    """)
    
    st.markdown("### 🔧 Configurações Técnicas")
    if st.checkbox("Mostrar detalhes técnicos"):
        st.json({
            "Modelo": "XGBoost Classifier",
            "Endpoint": ENDPOINT_NAME,
            "Região AWS": AWS_REGION,
            "Características": len(features) if 'features' in locals() else 0
        })