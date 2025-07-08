# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import os
from groq import Groq
from dotenv import load_dotenv
from io import BytesIO
import base64
from streamlit_echarts import st_echarts

# Chargement des variables d'environnement
load_dotenv()

# Configuration de la page
st.set_page_config(
    page_title="CornAI - Deep Consulting",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Style CSS amélioré
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700&family=Open+Sans:wght@400;600&display=swap');

    body, .stMarkdown, .stTextInput, .stButton {
        font-family: 'Open Sans', sans-serif;
        background: #f8f9fa;
    }

    .main-header, .feature-card, .disease-card, .cta-section, .results-section {
        animation: fadeIn 0.5s ease-in-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }

    .main-title {
        font-family: 'Montserrat', sans-serif;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }

    .main-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
    }

    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        text-align: center;
        height: 100%;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid #f0f0f0;
    }

    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }

    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }

    .feature-title {
        font-family: 'Montserrat', sans-serif;
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }

    .feature-description {
        color: #7f8c8d;
        line-height: 1.6;
    }

    .disease-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
    }

    .disease-section h2 {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 2rem;
        font-size: 2rem;
        font-family: 'Montserrat', sans-serif;
    }

    .disease-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        text-align: center;
        height: 100%;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .disease-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }

    .disease-image {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        margin: 0 auto 1rem;
        display: block;
        object-fit: cover;
        border: 3px solid #e9ecef;
        transition: transform 0.3s ease;
        cursor: pointer;
    }

    .disease-image:hover {
        transform: scale(1.1);
        border-color: #28a745;
    }

    .disease-color-brown { border-color: #8b4513; }
    .disease-color-red { border-color: #dc3545; }
    .disease-color-green { border-color: #28a745; }

    .disease-title {
        font-family: 'Montserrat', sans-serif;
        font-size: 1.3rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }

    .disease-description {
        color: #6c757d;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    .cta-section {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        padding: 3rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 2rem 0;
    }

    .cta-title {
        font-family: 'Montserrat', sans-serif;
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }

    .cta-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-bottom: 2rem;
    }

    .cta-button {
        background: white;
        color: #28a745;
        padding: 1rem 2rem;
        border: none;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .cta-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        background: #f8f9fa;
    }

    .upload-area {
        background: #f8f9ff;
        border: 2px dashed #28a745;
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
    }

    .upload-area:hover {
        border-color: #218838;
        background: #e9ecef;
    }

    .upload-area::before {
        content: "📤 Glissez votre image ici ou cliquez pour sélectionner";
        display: block;
        text-align: center;
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }

    .results-section {
        background: linear-gradient(145deg, #ffffff, #f1f3f5);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
    }

    .confidence-badge {
        background: linear-gradient(135deg, #764ba2, #4b0082);
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 0.5rem;
        border: 2px solid #4b0082;
        font-family: 'Montserrat', sans-serif;
        font-size: 1.5rem;
        font-weight: bold;
        transition: transform 0.3s ease;
    }

    .confidence-badge:hover {
        transform: scale(1.05);
    }

    .disease-name {
        text-align: center;
        font-family: 'Montserrat', sans-serif;
        font-size: 1.2rem;
        font-weight: bold;
        color: #2c3e50;
    }

    .recommendation-box {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }

    .recommendation-item {
        background: rgba(255,255,255,0.8);
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #e17055;
    }

    .chatbot-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 2rem;
    }

    .chatbot-messages {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 15px;
        padding: 1rem;
        max-height: 400px;
        overflow-y: auto;
    }

    .chat-message {
        display: flex;
        align-items: center;
        margin: 0.5rem;
        padding: 1rem;
        border-radius: 15px;
        max-width: 80%;
        position: relative;
    }

    .user-message {
        background: #667eea;
        color: white;
        margin-left: auto;
    }

    .user-message::before {
        content: "👨‍🌾";
        margin-right: 0.5rem;
    }

    .bot-message {
        background: #e9ecef;
        color: #2c3e50;
        margin-right: auto;
    }

    .bot-message::before {
        content: "🤖";
        margin-right: 0.5rem;
    }

    .typing-animation::after {
        content: "...";
        animation: typing 1s infinite;
    }

    @keyframes typing {
        0% { content: "."; }
        33% { content: ".."; }
        66% { content: "..."; }
    }

    .chat-input {
        display: flex;
        align-items: center;
    }

    .chat-input input {
        flex: 1;
        margin-right: 0.5rem;
    }

    .chat-input .send-icon {
        cursor: pointer;
        font-size: 1.2rem;
        color: #28a745;
    }

    .chat-input .send-icon:hover {
        color: #218838;
    }

    .nav-buttons {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
    }

    .nav-button {
        background: #28a745;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 0 0.25rem;
        cursor: pointer;
        font-weight: bold;
        transition: background 0.3s ease;
    }

    .nav-button:hover {
        background: #218838;
    }

    .nav-button.active {
        background: #667eea;
    }

    button[kind="primary"] {
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(40, 167, 69, 0); }
        100% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0); }
    }

    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        .feature-card, .disease-card {
            margin-bottom: 1rem;
        }
        .nav-buttons {
            position: static;
            display: flex;
            justify-content: center;
            margin-bottom: 1rem;
        }
        .upload-area {
            padding: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Dictionnaire des maladies du maïs avec images
DISEASES = {
    0: {
        "name": "Maïs Sain",
        "description": "Feuilles vertes et saines sans signes de maladie. Indicateur d'une bonne santé de la plante et de conditions de croissance optimales.",
        "symptoms": "Feuilles vertes uniformes, sans taches, nécrose ou stries.",
        "treatment": "Aucun traitement nécessaire.",
        "prevention": "Maintenir de bonnes pratiques agricoles : irrigation adéquate, rotation des cultures, semences certifiées.",
        "image": "images/sain_mais.jpg",
        "color": "green"
    },
    1: {
        "name": "Rouille du Maïs",
        "description": "Maladie fongique caractérisée par des pustules brun-orangé sur les feuilles. Elle peut réduire significativement le rendement si elle n'est pas traitée à temps.",
        "symptoms": "Nécrose des feuilles, décoloration, croissance ralentie, mort des tissus.",
        "treatment": "Détruire les plantes infectées (brûlage ou enfouissement), utiliser des semences résistantes.",
        "prevention": "Isoler les plantes affectées, éviter la replantation dans les zones infectées, collaborer avec les services agricoles.",
        "image": "images/mln_mais.jpg",
        "color": "brown"
    },
    2: {
        "name": "Brûlure des Feuilles",
        "description": "Infection causant des taches nécrotiques sur les feuilles, pouvant s'étendre rapidement dans des conditions d'humidité élevée.",
        "symptoms": "Stries ou rayures jaunes/blanches sur les feuilles, croissance réduite.",
        "treatment": "Contrôler les cicadelles (vecteurs) avec des insecticides, retirer les plantes infectées.",
        "prevention": "Planter des variétés résistantes, rotation des cultures avec des non-hôtes (ex. : légumineuses).",
        "image": "images/msv_mais.jpg",
        "color": "red"
    }
}

# Initialisation du client Groq
def init_groq_client():
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("⚠️ Clé API Groq manquante. Veuillez définir GROQ_API_KEY dans votre fichier .env")
            return None
        return Groq(api_key=api_key)
    except Exception as e:
        st.error(f"❌ Erreur d'initialisation Groq: {str(e)}")
        return None

# Charger le modèle PyTorch
@st.cache_resource
def load_maize_model():
    try:
        model_path = "best_model.pth"
        if not os.path.exists(model_path):
            st.error(f"❌ Modèle introuvable : {model_path}")
            return None
        
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 3)
        model = torch.nn.Sequential(model, torch.nn.Dropout(0.5))
        
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Erreur de chargement du modèle : {str(e)}")
        return None

# Fonction pour convertir une image en base64
def image_to_base64(image):
    try:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()
    except:
        return ""

# Fonction pour charger les images locales
def load_local_image(image_path):
    try:
        return Image.open(image_path)
    except:
        return Image.new('RGB', (80, 80), color='lightgray')

# Préparer l'image pour la prédiction
def prepare_image(image):
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img = image.convert('RGB')
        img = transform(img).unsqueeze(0)
        return img
    except Exception as e:
        st.error(f"❌ Erreur lors du traitement de l'image : {str(e)}")
        return None

# Générer des recommandations avec Groq
def generate_recommendations_with_groq(disease_name, confidence):
    groq_client = init_groq_client()
    if not groq_client:
        return generate_basic_recommendations(disease_name)
    
    try:
        prompt = f"""
        Tu es un expert agronome spécialisé dans les maladies du maïs en Afrique.
        
        Diagnostic: {disease_name}
        Niveau de confiance: {confidence:.1%}
        
        Génère 4 recommandations pratiques et spécifiques en français, sous forme de phrases courtes et actionables, basées sur la maladie prédite.
        Concentre-toi sur les actions immédiates à prendre par l'agriculteur pour traiter ou prévenir la maladie spécifique.
        Inclure des conseils adaptés au contexte africain, comme l'utilisation de ressources locales ou des pratiques agricoles durables.
        
        Format: une recommandation par ligne, sans numérotation ni puces.
        """
        
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            temperature=0.7,
            max_tokens=200
        )
        
        recommendations = response.choices[0].message.content.strip().split('\n')
        return [rec.strip() for rec in recommendations if rec.strip()][:4]
        
    except Exception as e:
        st.error(f"❌ Erreur API Groq pour les recommandations: {str(e)}")
        return generate_basic_recommendations(disease_name)

# Générer une réponse de chatbot avec Groq
def generate_chat_response_with_groq(user_message, context=""):
    groq_client = init_groq_client()
    if not groq_client:
        return "❌ Assistant IA temporairement indisponible. L'API Groq n'est pas accessible."
    
    try:
        prompt = f"""
        Tu es un assistant IA expert en maladies du maïs pour les agriculteurs africains.
        
        Contexte: {context}
        Question de l'utilisateur: {user_message}
        
        Réponds en français de manière claire, pratique et bienveillante.
        Donne des conseils spécifiques et actionables, adaptés au contexte africain.
        Maximum 150 mots.
        """
        
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            temperature=0.6,
            max_tokens=200
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        st.error(f"❌ Erreur API Groq pour le chatbot: {str(e)}")
        return "❌ Désolé, je ne peux pas répondre pour le moment. Veuillez réessayer."

# Recommandations de base (fallback)
def generate_basic_recommendations(disease_name):
    basic_recs = {
        "Maïs Sain": [
            "Maintenir les bonnes pratiques de culture actuelles",
            "Surveiller régulièrement l'état des plantations",
            "Assurer une irrigation équilibrée",
            "Continuer la rotation des cultures"
        ],
        "Rouille du Maïs": [
            "Appliquer un fongicide à base de soufre disponible localement",
            "Retirer et brûler les feuilles infectées pour limiter la propagation",
            "Améliorer la ventilation en espaçant les plants",
            "Collaborer avec les services agricoles pour des variétés résistantes"
        ],
        "Brûlure des Feuilles": [
            "Isoler les plants infectés pour éviter la propagation",
            "Utiliser des insecticides locaux contre les cicadelles",
            "Réduire l'humidité par un arrosage modéré",
            "Planter des variétés résistantes adaptées au climat local"
        ]
    }
    return basic_recs.get(disease_name, ["Consulter un expert agronome local"])

# Navigation
def show_navigation():
    st.markdown("""
    <div class="nav-buttons">
        <button class="nav-button {}" onclick="document.location.href='#accueil'">ACCUEIL</button>
        <button class="nav-button {}" onclick="document.location.href='#diagnostic'">DIAGNOSTIC</button>
        <span style="margin-left: 1rem;">🌐 EN</span>
    </div>
    """.format(
        "active" if st.session_state.get('current_page', 'accueil') == 'accueil' else "",
        "active" if st.session_state.get('current_page', 'accueil') == 'diagnostic' else ""
    ), unsafe_allow_html=True)

# Page d'accueil
def show_home_page():
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🌽 CornAI - Deep Consulting</div>
        <div class="main-subtitle">Détectez instantanément les maladies du maïs grâce à l'intelligence artificielle. Notre système avancé analyse vos images et fournit des recommandations personnalisées pour protéger vos cultures.</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">Analyse Rapide</div>
            <div class="feature-description">Obtenez des résultats en quelques secondes grâce à notre algorithme d'intelligence artificielle optimisé pour la reconnaissance des maladies du maïs.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Précision Élevée</div>
            <div class="feature-description">Notre modèle entraîné sur des milliers d'images garantit une précision supérieure à 95% dans la détection des maladies les plus courantes.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">💬</div>
            <div class="feature-title">Assistant IA</div>
            <div class="feature-description">Posez vos questions à notre assistant intelligent pour obtenir des conseils personnalisés sur la gestion et la prévention des maladies.</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="disease-section">
        <h2>Classes de Maladies Détectées</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    for i, (disease_id, disease_info) in enumerate(DISEASES.items()):
        target_col = col1 if i == 0 else col2 if i == 1 else col3
        
        with target_col:
            disease_image = load_local_image(disease_info['image'])
            image_b64 = image_to_base64(disease_image)
            
            st.markdown(f"""
            <div class="disease-card">
                <img src="data:image/jpeg;base64,{image_b64}" class="disease-image disease-color-{disease_info['color']}" alt="{disease_info['name']}" title="{disease_info['description']}">
                <div class="disease-title">{disease_info['name']}</div>
                <div class="disease-description">{disease_info['description']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="cta-section">
        <div class="cta-title">Prêt à Commencer ?</div>
        <div class="cta-subtitle">Téléchargez une image de votre plant de maïs et obtenez un diagnostic instantané.</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("COMMENCER LE DIAGNOSTIC", key="start_diagnostic", use_container_width=True):
            st.session_state.current_page = 'diagnostic'
            st.rerun()

# Page de diagnostic
def show_diagnostic_page():
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🌽 Diagnostic Maïs IA</div>
        <div class="main-subtitle">Détection intelligente des maladies du maïs par analyse d'image</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🡸 Retour à l'accueil", key="back_home"):
        st.session_state.current_page = 'accueil'
        st.rerun()
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📁 Importer une image")
        
        uploaded_file = st.file_uploader(
            "Cliquez ici ou glissez-déposez votre image",
            type=["jpg", "jpeg", "png"],
            help="Formats acceptés: JPG, PNG, JPEG (max 5MB)"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Feuille analysée", use_column_width=True)
            
            if st.button("🔍 ANALYSER L'IMAGE", key="analyze", use_container_width=True, type="primary"):
                with st.spinner("🤖 Analyse en cours..."):
                    model = load_maize_model()
                    
                    if model:
                        img_array = prepare_image(image)
                        
                        if img_array is not None:
                            try:
                                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                model.to(device)
                                img_array = img_array.to(device)
                                
                                with torch.no_grad():
                                    output = model(img_array)
                                    probabilities = torch.softmax(output, dim=1)[0]
                                    _, predicted = torch.max(output, 1)
                                    class_idx = predicted.item()
                                    confidence = probabilities[class_idx].item()
                                    disease_name = DISEASES[class_idx]["name"]
                                    
                                    st.session_state.results = {
                                        "disease": disease_name,
                                        "confidence": confidence,
                                        "probabilities": probabilities.cpu().numpy(),
                                        "recommendations": generate_recommendations_with_groq(disease_name, confidence)
                                    }
                                    
                                    st.success("✅ Analyse terminée avec succès!")
                                    
                            except Exception as e:
                                st.error(f"❌ Erreur d'analyse : {str(e)}")
    
    with col2:
        st.markdown("### 📊 Résultats de l'analyse")
        
        if "results" in st.session_state:
            results = st.session_state.results
            
            st.markdown(f"""
            <div class="confidence-badge">
                <h3>{results['confidence']:.0%}</h3>
            </div>
            <div class="disease-name">{results['disease']}</div>
            """, unsafe_allow_html=True)
            
            # Graphique circulaire des probabilités
            st.markdown("### 📊 Probabilités de prédiction")
            probabilities = [float(p) for p in results['probabilities']]  # Convert numpy.float32 to float
            labels = ["Maïs Sain", "Rouille du Maïs", "Brûlure des Feuilles"]
            colors = ["#28a745", "#8b4513", "#dc3545"]
            
            options = {
                "tooltip": {"trigger": "item"},
                "legend": {
                    "top": "5%",
                    "left": "center",
                    "textStyle": {"fontFamily": "'Open Sans', sans-serif", "fontSize": 14}
                },
                "series": [
                    {
                        "name": "Probabilités",
                        "type": "pie",
                        "radius": ["40%", "70%"],
                        "avoidLabelOverlap": False,
                        "label": {"show": False, "position": "center"},
                        "emphasis": {
                            "label": {"show": True, "fontSize": 16, "fontWeight": "bold"}
                        },
                        "labelLine": {"show": False},
                        "data": [
                            {"value": probabilities[0], "name": labels[0], "itemStyle": {"color": colors[0]}},
                            {"value": probabilities[1], "name": labels[1], "itemStyle": {"color": colors[1]}},
                            {"value": probabilities[2], "name": labels[2], "itemStyle": {"color": colors[2]}}
                        ]
                    }
                ]
            }
            st_echarts(options=options, height="400px")
            
            st.markdown("### 🎯 Recommandations")
            st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
            
            for i, rec in enumerate(results['recommendations']):
                st.markdown(f"""
                <div class="recommendation-item">
                    {rec}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("### 🤖 Assistant IA")
            st.markdown('<div class="chatbot-section">', unsafe_allow_html=True)
            
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = [
                    {"role": "bot", "content": "Bonjour ! Je suis votre assistant IA spécialisé dans les maladies du maïs. Comment puis-je vous aider aujourd'hui ?"}
                ]
            
            st.markdown('<div class="chatbot-messages">', unsafe_allow_html=True)
            for message in st.session_state.chat_history:
                st.markdown(f"""
                <div class="chat-message {message['role']}-message">
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("🗑️ Vider l'historique du chat", key="clear_chat"):
                st.session_state.chat_history = [
                    {"role": "bot", "content": "Bonjour ! Je suis votre assistant IA spécialisé dans les maladies du maïs. Comment puis-je vous aider aujourd'hui ?"}
                ]
                st.rerun()
            
            st.markdown('<div class="chat-input">', unsafe_allow_html=True)
            user_input = st.text_input("", key="chat_input", placeholder="C'est quoi les meilleures techniques culturales en afrique subsaharienne")
            if st.button("➡️", key="send_icon", help="Envoyer (ou appuyez sur Entrée)"):
                if user_input:
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    context = f"Dernière analyse: {st.session_state.results['disease']}"
                    with st.spinner("🤖 Réponse en cours..."):
                        bot_response = generate_chat_response_with_groq(user_input, context)
                        st.session_state.chat_history.append({"role": "bot", "content": bot_response})
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            st.info("👆 Téléchargez une image et cliquez sur 'Analyser' pour voir les résultats")

# Navigation principale
def main():
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'accueil'
    
    show_navigation()
    
    if st.session_state.current_page == 'accueil':
        show_home_page()
    else:
        show_diagnostic_page()

if __name__ == "__main__":
    main()