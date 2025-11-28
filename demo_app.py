import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import json

st.title("🔒 Network Intrusion Detection System")

try:
    # Load model with custom objects to handle Dropout layers
    custom_objects = {
        'Dropout': tf.keras.layers.Dropout
    }
    
    model = tf.keras.models.load_model('ids_model.h5', 
                                     compile=False, 
                                     custom_objects=custom_objects)
    
    # Manually compile
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    scaler = joblib.load('scaler.pkl')
    
    with open('model_data.json', 'r') as f:
        data = json.load(f)
    
    st.success("✅ Model loaded successfully!")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "📖 User Guide", "🔍 Manual Test", "🧪 Sample Test"])
    
    with tab1:
        st.header("Network Intrusion Detection System")
        st.write(f"**Input Features:** {data['input_features']}")
        st.write(f"**Detection Threshold:** {data['threshold']:.6f}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Type", "Autoencoder")
        with col2:
            st.metric("Dataset", "CICIDS2017")
        with col3:
            st.metric("Status", "🟢 Active")
    
    with tab2:
        st.header("📖 User Guide - Network Flow Features")
        
        st.markdown("""
        ### What is Network Flow Analysis?
        Network flows represent communication between devices. Each flow has characteristics that help identify normal vs malicious behavior.
        
        ### Key Network Features Explained:
        """)
        
        # Feature explanations
        feature_guide = {
            "Flow Duration": {
                "description": "How long the connection lasted",
                "normal_range": "100-5000 ms",
                "attack_indicator": "> 10000 ms (very long connections)",
                "example_normal": "2000",
                "example_attack": "15000"
            },
            "Packet Count": {
                "description": "Number of data packets sent",
                "normal_range": "5-50 packets",
                "attack_indicator": "> 100 packets (flooding)",
                "example_normal": "25",
                "example_attack": "500"
            },
            "Bytes per Second": {
                "description": "Data transfer rate",
                "normal_range": "100-2000 bytes/sec",
                "attack_indicator": "> 5000 bytes/sec (data exfiltration)",
                "example_normal": "800",
                "example_attack": "10000"
            },
            "Packet Size Average": {
                "description": "Average size of each packet",
                "normal_range": "40-1500 bytes",
                "attack_indicator": "> 2000 bytes (unusual payload)",
                "example_normal": "512",
                "example_attack": "3000"
            },
            "Connection Frequency": {
                "description": "How often connections are made",
                "normal_range": "1-10 per minute",
                "attack_indicator": "> 50 per minute (scanning)",
                "example_normal": "5",
                "example_attack": "100"
            }
        }
        
        for feature, info in feature_guide.items():
            with st.expander(f"📊 {feature}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Description:** {info['description']}")
                    st.write(f"**Normal Range:** {info['normal_range']}")
                    st.success(f"**Normal Example:** {info['example_normal']}")
                with col2:
                    st.write(f"**Attack Indicator:** {info['attack_indicator']}")
                    st.error(f"**Attack Example:** {info['example_attack']}")
        
        st.markdown("""
        ### 🎯 Quick Testing Tips:
        
        **For Normal Traffic:**
        - Use moderate values (not too high/low)
        - Flow Duration: 1000-3000
        - Packet Count: 10-30
        - Bytes/sec: 500-1500
        
        **For Attack Simulation:**
        - Use extreme values
        - Flow Duration: > 10000
        - Packet Count: > 100
        - Bytes/sec: > 5000
        """)
    
    with tab3:
        st.header("🔍 Manual Network Flow Testing")
        
        st.markdown("### Enter Network Flow Characteristics:")
        st.info("💡 Tip: Check the User Guide tab for feature explanations and example values!")
        
        # Simplified input fields with guidance
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Basic Flow Metrics")
            flow_duration = st.number_input(
                "Flow Duration (milliseconds)", 
                value=2000.0, 
                help="How long the connection lasted. Normal: 1000-5000, Suspicious: >10000"
            )
            
            packet_count = st.number_input(
                "Total Packets", 
                value=25.0,
                help="Number of packets sent. Normal: 10-50, Suspicious: >100"
            )
            
            bytes_per_sec = st.number_input(
                "Bytes per Second", 
                value=800.0,
                help="Data transfer rate. Normal: 500-2000, Suspicious: >5000"
            )
        
        with col2:
            st.subheader("Advanced Metrics")
            packet_size = st.number_input(
                "Average Packet Size (bytes)", 
                value=512.0,
                help="Average packet size. Normal: 64-1500, Suspicious: >2000"
            )
            
            connection_freq = st.number_input(
                "Connections per Minute", 
                value=5.0,
                help="Connection frequency. Normal: 1-10, Suspicious: >50"
            )
            
            protocol_factor = st.selectbox(
                "Protocol Type", 
                ["TCP (1.0)", "UDP (1.5)", "ICMP (2.0)"],
                help="Protocol multiplier for analysis"
            )
        
        # Generate remaining features
        protocol_multiplier = float(protocol_factor.split("(")[1].split(")")[0])
        
        # Create feature array (simplified to 10 key features)
        features = [
            flow_duration,
            packet_count, 
            bytes_per_sec,
            packet_size,
            connection_freq * protocol_multiplier,
            flow_duration / packet_count if packet_count > 0 else 0,  # Time per packet
            packet_size * packet_count,  # Total bytes
            bytes_per_sec / packet_size if packet_size > 0 else 0,  # Packets per second
            np.log(flow_duration + 1),  # Log duration
            packet_count / (flow_duration / 1000) if flow_duration > 0 else 0  # Packet rate
        ]
        
        # Pad with zeros to match model input size
        while len(features) < data['input_features']:
            features.append(0.0)
        
        # Prediction
        if st.button("🔍 Analyze Network Flow", type="primary"):
            def predict_intrusion(features):
                scaled = scaler.transform([features])
                pred = model.predict(scaled, verbose=0)
                error = np.mean((scaled - pred) ** 2)
                is_attack = error > data['threshold']
                confidence = error / data['threshold']
                return is_attack, confidence, error
            
            is_attack, confidence, error = predict_intrusion(features)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if is_attack:
                    st.error("🚨 POTENTIAL INTRUSION DETECTED!")
                    st.markdown("**Risk Level:** HIGH")
                    
                    # Explain why it's suspicious
                    suspicious_factors = []
                    if flow_duration > 8000:
                        suspicious_factors.append("⚠️ Unusually long connection duration")
                    if packet_count > 80:
                        suspicious_factors.append("⚠️ High packet count (possible flooding)")
                    if bytes_per_sec > 4000:
                        suspicious_factors.append("⚠️ High data transfer rate")
                    if packet_size > 1800:
                        suspicious_factors.append("⚠️ Large packet size")
                    if connection_freq > 30:
                        suspicious_factors.append("⚠️ High connection frequency")
                    
                    if suspicious_factors:
                        st.markdown("**Suspicious Indicators:**")
                        for factor in suspicious_factors:
                            st.markdown(f"- {factor}")
                else:
                    st.success("✅ NORMAL NETWORK TRAFFIC")
                    st.markdown("**Risk Level:** LOW")
                    st.markdown("All network characteristics appear normal.")
            
            with col2:
                st.subheader("Technical Details")
                st.metric("Confidence Score", f"{confidence:.2f}")
                st.metric("Reconstruction Error", f"{error:.6f}")
                st.metric("Detection Threshold", f"{data['threshold']:.6f}")
                
                # Confidence explanation
                if confidence < 0.8:
                    st.info("🔵 Very confident this is normal traffic")
                elif confidence < 1.2:
                    st.warning("🟡 Borderline - monitor closely")
                else:
                    st.error("🔴 High confidence this is malicious")
    
    with tab4:
        st.header("🧪 Sample Data Testing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Normal Traffic Sample")
            st.info("This sample represents typical web browsing traffic")
            
            if st.button("🔍 Test Normal Traffic Sample"):
                def predict_intrusion(features):
                    scaled = scaler.transform([features])
                    pred = model.predict(scaled, verbose=0)
                    error = np.mean((scaled - pred) ** 2)
                    is_attack = error > data['threshold']
                    confidence = error / data['threshold']
                    return is_attack, confidence, error
                
                is_attack, confidence, error = predict_intrusion(data['sample_benign'])
                
                if is_attack:
                    st.error("❌ False Positive - Normal traffic flagged as attack")
                else:
                    st.success("✅ Correctly Identified as Normal Traffic")
                
                st.write(f"**Confidence:** {confidence:.2f}")
                st.write(f"**Error:** {error:.6f}")
        
        with col2:
            st.subheader("Attack Traffic Sample")
            st.warning("This sample represents malicious network activity")
            
            if st.button("🚨 Test Attack Traffic Sample"):
                def predict_intrusion(features):
                    scaled = scaler.transform([features])
                    pred = model.predict(scaled, verbose=0)
                    error = np.mean((scaled - pred) ** 2)
                    is_attack = error > data['threshold']
                    confidence = error / data['threshold']
                    return is_attack, confidence, error
                
                is_attack, confidence, error = predict_intrusion(data['sample_attack'])
                
                if is_attack:
                    st.success("✅ Attack Successfully Detected!")
                else:
                    st.error("❌ False Negative - Attack missed")
                
                st.write(f"**Confidence:** {confidence:.2f}")
                st.write(f"**Error:** {error:.6f}")

except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.write("Make sure these files are present:")
    st.write("- ids_model.h5")
    st.write("- scaler.pkl") 
    st.write("- model_data.json")
