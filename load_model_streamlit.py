import streamlit as st
import pandas as pd
import joblib
import json
import requests
from datetime import datetime
import os
from config import *

# Configure page
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=PAGE_LAYOUT,
    initial_sidebar_state="expanded"
)

# Load saved objects
@st.cache_resource
def load_models():
    model = joblib.load(MODEL_FILES["model"])
    scaler = joblib.load(MODEL_FILES["scaler"])
    le = joblib.load(MODEL_FILES["label_encoder"])
    X_columns = joblib.load(MODEL_FILES["X_columns"])
    return model, scaler, le, X_columns

model, scaler, le, X_columns = load_models()

def predict_life_satisfaction(user_input: dict):
    input_df = pd.DataFrame([user_input])
    input_encoded = pd.get_dummies(input_df).reindex(columns=X_columns, fill_value=0)
    input_scaled = scaler.transform(input_encoded)
    pred = model.predict(input_scaled)
    return le.inverse_transform(pred)[0]

def get_prediction_probabilities(user_input: dict):
    input_df = pd.DataFrame([user_input])
    input_encoded = pd.get_dummies(input_df).reindex(columns=X_columns, fill_value=0)
    input_scaled = scaler.transform(input_encoded)
    
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_scaled)[0]
        proba_dict = {}
        for i, p in enumerate(proba):
            label = le.inverse_transform([i])[0]
            if label.lower() != "unknown":
                proba_dict[label] = p
        return proba_dict
    return None

def call_deepseek_api(messages):
    """Call DeepSeek API for chatbot responses"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": CHATBOT_MODEL,
        "messages": messages,
        "temperature": CHATBOT_TEMPERATURE,
        "max_tokens": CHATBOT_MAX_TOKENS
    }
    
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            if USE_FALLBACK_RESPONSES:
                return get_fallback_response(messages[-1]["content"])
            else:
                return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        if USE_FALLBACK_RESPONSES:
            return get_fallback_response(messages[-1]["content"])
        else:
            return f"Error connecting to AI service: {str(e)}"

def get_fallback_response(user_message):
    """Provide helpful fallback responses when API is not available"""
    user_message = user_message.lower()
    
    # Life satisfaction prediction explanations
    if "prediction" in user_message and "mean" in user_message:
        return """Your life satisfaction prediction is based on multiple factors including health, relationships, work, and lifestyle. The model analyzes your responses across these dimensions to provide an overall assessment. Higher scores generally indicate better life satisfaction, while lower scores suggest areas for improvement."""
    
    elif "improve" in user_message or "better" in user_message:
        return """To improve your life satisfaction, focus on these key areas:
1. **Health**: Regular exercise, good sleep, and healthy eating
2. **Social Connections**: Strengthen relationships with family and friends
3. **Purpose**: Set meaningful goals and pursue activities you enjoy
4. **Mindfulness**: Practice stress management and gratitude
5. **Work-Life Balance**: Find satisfaction in your career while maintaining personal time"""
    
    elif "health" in user_message:
        return """Physical and mental health are crucial for life satisfaction. Consider:
- Regular exercise (30 minutes daily)
- Balanced nutrition
- 7-9 hours of quality sleep
- Stress management techniques
- Regular health check-ups
- Seeking professional help when needed"""
    
    elif "social" in user_message or "friends" in user_message or "family" in user_message:
        return """Strong social connections significantly impact life satisfaction:
- Spend quality time with family and friends
- Join community groups or clubs
- Volunteer for causes you care about
- Practice active listening and empathy
- Maintain regular communication with loved ones
- Build new relationships through shared interests"""
    
    elif "work" in user_message or "job" in user_message or "career" in user_message:
        return """Work satisfaction contributes to overall life satisfaction:
- Find meaning in your work or consider career changes
- Develop new skills and pursue growth opportunities
- Maintain work-life balance
- Build positive relationships with colleagues
- Set achievable goals and celebrate achievements
- Consider if your current role aligns with your values"""
    
    elif "stress" in user_message or "anxiety" in user_message or "worry" in user_message:
        return """Managing stress is essential for life satisfaction:
- Practice deep breathing and meditation
- Exercise regularly to reduce stress hormones
- Maintain a regular sleep schedule
- Set boundaries and learn to say no
- Seek professional counseling if needed
- Focus on what you can control"""
    
    elif "happiness" in user_message or "happy" in user_message:
        return """Happiness and life satisfaction are closely related:
- Practice gratitude daily
- Engage in activities you enjoy
- Help others through volunteering
- Spend time in nature
- Develop meaningful relationships
- Pursue personal growth and learning
- Focus on the present moment"""
    
    else:
        return """I'm here to help you understand your life satisfaction and provide guidance for improvement. You can ask me about:
- What your prediction means
- How to improve your life satisfaction
- Health and wellness tips
- Social connection strategies
- Work and career advice
- Stress management techniques
- General happiness and wellbeing

Feel free to ask any specific questions about these topics!"""

def get_personalized_recommendations(prediction, user_input, confidence):
    """Generate personalized recommendations based on prediction and user input"""
    
    recommendations = {
        "Very Low": {
            "health": "Consider consulting a healthcare professional for mental health support",
            "social": "Try joining community groups or social activities to build connections",
            "lifestyle": "Establish a daily routine with regular sleep and exercise",
            "professional": "Consider career counseling or skill development programs"
        },
        "Low": {
            "health": "Focus on regular exercise and healthy eating habits",
            "social": "Reach out to friends and family more frequently",
            "lifestyle": "Practice stress management techniques like meditation",
            "professional": "Set small, achievable goals at work"
        },
        "Medium": {
            "health": "Maintain your current health routine and consider new activities",
            "social": "Strengthen existing relationships and try new social activities",
            "lifestyle": "Explore new hobbies or interests",
            "professional": "Look for opportunities to grow in your career"
        },
        "High": {
            "health": "Keep up your excellent health habits",
            "social": "Continue nurturing your strong social connections",
            "lifestyle": "Share your positive energy with others",
            "professional": "Mentor others or take on leadership roles"
        },
        "Very High": {
            "health": "You're doing great! Consider helping others with their health goals",
            "social": "Your positive attitude is valuable - spread it to others",
            "lifestyle": "Document your successful strategies to help others",
            "professional": "Your satisfaction can inspire workplace improvements"
        }
    }
    
    return recommendations.get(prediction, recommendations["Medium"])

def save_user_progress(user_input, prediction, confidence, recommendations):
    """Save user progress to a JSON file"""
    progress_data = {
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "prediction": prediction,
        "confidence": confidence,
        "recommendations": recommendations
    }
    
    # Create progress directory if it doesn't exist
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    
    # Save to file
    filename = f"{PROGRESS_DIR}/{PROGRESS_FILE_PREFIX}{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(progress_data, f, indent=2)
    
    return filename

# Category options from dataset
categories = {
    "A2": ['Average', "Don't know", 'Poor', 'Very poor', 'Very well', 'Well'],
    "C1": ["Don't know", 'No', 'Yes'],
    "D2": ['Always', "Don't know", 'Never', 'Often', 'Rarely', 'Refuse to answer', 'Sometimes'],
    "D4": ['Always', "Don't know", 'Never', 'Often', 'Rarely', 'Sometimes'],
    "D8": ['Always', "Don't know", 'Never', 'Often', 'Rarely', 'Sometimes'],
    "D10": ['Always', "Don't know", 'Never', 'Often', 'Rarely', 'Refuse to answer', 'Sometimes'],
    "D15": ['Always', "Don't know", 'Never', 'Often', 'Rarely', 'Refuse to answer', 'Sometimes'],
    "M8": ['Average', 'Bad', "Don't Know", 'Good', 'Refuse to answer', 'Very bad', 'Very good'],
    "E17": ['Children', "Don't know", 'Friends/colleagues',
            "I don't share this with anyone", 'Other family', 'Others',
            'Parents', 'Partner/spouse/boy-/girlfriend',
            'Refuse to answer', 'Siblings', 'Staff'],
    "G1": ['No', 'Refuse to answer', 'Yes'],
    "J2": ['Daily', 'Less frequently', 'Never', 'Once a month', 'Once a week',
           'Refuse to answer', 'Several times a month', 'Several times a week'],
    "J4": ['Daily', 'Less frequently', 'Never', 'Once a month', 'Once a week',
           'Refuse to answer', 'Several times a month', 'Several times a week'],
    "J9": ['Daily', 'Less frequently', 'Never', 'Once a month', 'Once a week',
           'Several times a month', 'Several times a week'],
    "J17": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,23,24,25,26,27,
            30,31,40,45,50,52,100,999]
}

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None
if 'current_confidence' not in st.session_state:
    st.session_state.current_confidence = None
if 'current_recommendations' not in st.session_state:
    st.session_state.current_recommendations = None
if 'current_user_input' not in st.session_state:
    st.session_state.current_user_input = None

# Main UI
st.title("🌱 Life Satisfaction AI Assistant")
st.markdown("### Your Personal AI Guide to Life Satisfaction")

# Create tabs for different features
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Life Satisfaction Assessment",
    "🤖 AI Chatbot",
    "📈 Progress & Recommendations",
    "🔍 Explainable AI"
])


with tab1:
    st.header("Complete Your Life Satisfaction Assessment")
    st.write("Answer the following questions to get your personalized life satisfaction prediction and AI-powered insights.")
    
    # Create form for better UX
    with st.form("life_satisfaction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal Information")
            user_input = {}
            user_input["age"] = st.number_input("1. What is your age?", min_value=0, max_value=120, step=1)
            user_input["E1"] = st.number_input("2. How tall are you (cm)?", min_value=50.0, max_value=250.0, step=0.1)
            user_input["E2"] = st.number_input("3. How much do you weigh (kg)?", min_value=20.0, max_value=200.0, step=0.1)
            
        with col2:
            st.subheader("Health & Wellbeing")
            user_input["A2"] = st.selectbox("4. How would you rate your health generally?", categories["A2"])
            user_input["C1"] = st.selectbox("5. Do you suffer from a long-term physical health problem or disability?", categories["C1"])
            user_input["M6"] = st.number_input("6. In the past year, how much have you spent on medicine/supplements?", min_value=0.0, step=1.0)
        
        st.subheader("Mental Health & Emotional Wellbeing")
        col3, col4 = st.columns(2)
        
        with col3:
            user_input["D2"] = st.selectbox("7. Are you depressed?", categories["D2"])
            user_input["D4"] = st.selectbox("8. Do you often feel lonely?", categories["D4"])
            user_input["D8"] = st.selectbox("9. Do you see yourself as a person who worries a lot?", categories["D8"])
            
        with col4:
            user_input["D10"] = st.selectbox("10. Do you see yourself as emotionally stable?", categories["D10"])
            user_input["D15"] = st.selectbox("11. Do you prepare plans and implement them?", categories["D15"])
        
        st.subheader("Work & Finances")
        col5, col6 = st.columns(2)
        
        with col5:
            job_choice = st.selectbox("12. Do you have a job?", ["Yes", "No"])

            if job_choice == "Yes":
                user_input["job"] = "Holds an ordinary or supported job"
                user_input["F15"] = st.slider(
                    "13. On a scale from 0-10, how content are you with your job?",
                    0, 10, 5
                )
            else:
                user_input["job"] = "Doesn't hold an ordinary or supported job"
                user_input["F15"] = 0

                
        with col6:
            user_input["M8"] = st.selectbox("14. How would you rate your current finances?", categories["M8"])
        
        st.subheader("Social Life & Relationships")
        col7, col8 = st.columns(2)
        
        with col7:
            user_input["E17"] = st.selectbox("15. Who do you primarily talk to about personal and serious problems?", categories["E17"])
            user_input["G1"] = st.selectbox("16. Do you have a spouse/partner/boy/girlfriend?", categories["G1"])
            
        with col8:
            user_input["J2"] = st.selectbox("17. How often have you spent time with other relatives in the past year?", categories["J2"])
            user_input["J4"] = st.selectbox("18. In the past year, how often have you spent time with acquaintances?", categories["J4"])
        
        st.subheader("Lifestyle & Activities")
        col9, col10 = st.columns(2)
        
        with col9:
            user_input["J9"] = st.selectbox("19. In the past year, how often have you been to the cinema, a concert, or the theater?", categories["J9"])
            user_input["J17"] = st.selectbox("20. How often have you been abroad on holiday/family visits in the past year?", categories["J17"])
        
        submitted = st.form_submit_button("🚀 Get My Life Satisfaction Prediction", type="primary")
        
        if submitted:
            with st.spinner("Analyzing your responses..."):
                # Get prediction
                prediction = predict_life_satisfaction(user_input)
                proba_dict = get_prediction_probabilities(user_input)
                confidence = proba_dict[prediction] if proba_dict else None
                
                # Get recommendations
                recommendations = get_personalized_recommendations(prediction, user_input, confidence)
                
                # Store in session state
                st.session_state.current_prediction = prediction
                st.session_state.current_confidence = confidence
                st.session_state.current_recommendations = recommendations
                st.session_state.current_user_input = user_input
                
                # Display results
                st.success("✅ Analysis Complete!")
                
                col_result1, col_result2 = st.columns([2, 1])

                with col_result1:
                    st.metric("Life Satisfaction Level", prediction)


                with col_result2:
                    st.write("### Quick Insights")
                    if prediction in ["Very Low", "Low"]:
                        st.warning("Consider focusing on mental health and social connections")
                    elif prediction in ["Medium"]:
                        st.info("You're on a good path - small improvements can make a big difference")
                    else:
                        st.success("Excellent! Keep up your positive lifestyle choices")

    # Save progress button (completely outside the form)
    if st.session_state.current_prediction:
        if st.button("💾 Save My Results"):
            # Get the current values from session state
            prediction = st.session_state.current_prediction
            confidence = st.session_state.current_confidence
            recommendations = st.session_state.current_recommendations
            user_input = st.session_state.current_user_input
            filename = save_user_progress(user_input, prediction, confidence, recommendations)
            st.success(f"Results saved! File: {filename}")

with tab2:
    st.header("🤖 AI Chatbot Assistant")
    st.write("Chat with our AI assistant to understand your results, get personalized advice, or ask questions about life satisfaction.")
    
    # Chat interface
    if st.session_state.current_prediction:
        st.info(f"💡 I can help explain your current prediction: **{st.session_state.current_prediction}**")
    
    # Chat input
    user_message = st.chat_input("Ask me anything about life satisfaction, your results, or get personalized advice...")
    
    if user_message:
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": user_message})
        
        # Prepare context for AI
        context = f"""
        You are a helpful AI assistant specializing in life satisfaction and mental wellbeing. 
        
        Current user context:
        - Life satisfaction prediction: {st.session_state.current_prediction if st.session_state.current_prediction else 'Not available'}
        - Prediction confidence: {f"{st.session_state.current_confidence:.1%}" if st.session_state.current_confidence else 'Not available'}
        
        Your role:
        1. Explain life satisfaction concepts in simple terms
        2. Help users understand their prediction results
        3. Provide personalized advice based on their situation
        4. Guide users through the assessment process
        5. Offer practical tips for improving life satisfaction
        6. Be supportive, empathetic, and professional
        
        Keep responses conversational, helpful, and under 200 words unless the user asks for more detail.
        """
        
        # Prepare messages for API
        messages = [{"role": "system", "content": context}]
        messages.extend(st.session_state.chat_history)
        
        # Get AI response
        with st.spinner("AI is thinking..."):
            ai_response = call_deepseek_api(messages)
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
    
    # Quick action buttons
    if st.session_state.current_prediction:
        st.write("### Quick Questions")
        col_q1, col_q2, col_q3 = st.columns(3)
        
        with col_q1:
            if st.button("What does my prediction mean?"):
                # Prepare context for AI
                context = f"""
                You are a helpful AI assistant specializing in life satisfaction and mental wellbeing. 
                
                Current user context:
                - Life satisfaction prediction: {st.session_state.current_prediction if st.session_state.current_prediction else 'Not available'}
                - Prediction confidence: {f"{st.session_state.current_confidence:.1%}" if st.session_state.current_confidence else 'Not available'}
                
                Your role:
                1. Explain life satisfaction concepts in simple terms
                2. Help users understand their prediction results
                3. Provide personalized advice based on their situation
                4. Guide users through the assessment process
                5. Offer practical tips for improving life satisfaction
                6. Be supportive, empathetic, and professional
                
                Keep responses conversational, helpful, and under 200 words unless the user asks for more detail.
                """
                st.session_state.chat_history.append({"role": "user", "content": "What does my prediction mean?"})
                messages = [{"role": "system", "content": context}] + st.session_state.chat_history
                ai_response = call_deepseek_api(messages)
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                st.rerun()
        
        with col_q2:
            if st.button("How can I improve?"):
                # Prepare context for AI
                context = f"""
                You are a helpful AI assistant specializing in life satisfaction and mental wellbeing. 
                
                Current user context:
                - Life satisfaction prediction: {st.session_state.current_prediction if st.session_state.current_prediction else 'Not available'}
                - Prediction confidence: {f"{st.session_state.current_confidence:.1%}" if st.session_state.current_confidence else 'Not available'}
                
                Your role:
                1. Explain life satisfaction concepts in simple terms
                2. Help users understand their prediction results
                3. Provide personalized advice based on their situation
                4. Guide users through the assessment process
                5. Offer practical tips for improving life satisfaction
                6. Be supportive, empathetic, and professional
                
                Keep responses conversational, helpful, and under 200 words unless the user asks for more detail.
                """
                st.session_state.chat_history.append({"role": "user", "content": "How can I improve my life satisfaction?"})
                messages = [{"role": "system", "content": context}] + st.session_state.chat_history
                ai_response = call_deepseek_api(messages)
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                st.rerun()
        
        with col_q3:
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

with tab3:
    st.header("📈 Your Progress & Recommendations")
    
    if st.session_state.current_prediction:
        st.success(f"Latest Assessment: **{st.session_state.current_prediction}**")
        
        # Display detailed recommendations
        st.subheader("🎯 Personalized Recommendations")
        
        if st.session_state.current_recommendations:
            rec = st.session_state.current_recommendations
            
            col_rec1, col_rec2 = st.columns(2)
            
            with col_rec1:
                st.write("#### 💪 Health & Wellness")
                st.info(rec["health"])
                
                st.write("#### 👥 Social Connections")
                st.info(rec["social"])
            
            with col_rec2:
                st.write("#### 🌟 Lifestyle & Hobbies")
                st.info(rec["lifestyle"])
                
                st.write("#### 💼 Professional Growth")
                st.info(rec["professional"])
        
        # Progress tracking
        st.subheader("📊 Track Your Progress")
        
        # Check for existing progress files
        if os.path.exists(PROGRESS_DIR):
            progress_files = [f for f in os.listdir(PROGRESS_DIR) if f.endswith('.json')]
            
            if progress_files:
                st.write("Your previous assessments:")
                
                for i, filename in enumerate(sorted(progress_files, reverse=True)[:MAX_PROGRESS_FILES_DISPLAY]):  # Show last 5
                    filepath = os.path.join(PROGRESS_DIR, filename)
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                        
                        with st.expander(f"Assessment {i+1} - {data['timestamp'][:10]}"):
                            st.write(f"**Prediction:** {data['prediction']}")
                            st.write(f"**Confidence:** {data['confidence']:.1%}")
                            st.write(f"**Date:** {data['timestamp']}")
                    except:
                        st.write(f"Could not load {filename}")
            else:
                st.info("No previous assessments found. Complete your first assessment to start tracking!")
        else:
            st.info("No previous assessments found. Complete your first assessment to start tracking!")
    
    else:
        st.info("Complete your life satisfaction assessment in the first tab to see personalized recommendations and track your progress!")


with tab4:
    st.header("🔍 Explainable AI Insights")
    st.write("Understand how the model makes predictions using SHAP and LIME.")

    # === Column-to-Question Mapping ===
    with st.expander("📖 See which columns correspond to which questions"):
        st.markdown("""
        Here’s how the dataset columns map to the survey questions you answered:
        """)
        
        col_map = {
            "age": "1. What is your age?",
            "E1": "2. How tall are you (cm)?",
            "E2": "3. How much do you weigh (kg)?",
            "A2": "4. How would you rate your health generally?",
            "C1": "5. Do you suffer from a long-term physical health problem or disability?",
            "M6": "6. In the past year, how much have you spent on medicine/supplements?",
            "D2": "7. Are you depressed?",
            "D4": "8. Do you often feel lonely?",
            "D8": "9. Do you see yourself as a person who worries a lot?",
            "D10": "10. Do you see yourself as emotionally stable?",
            "D15": "11. Do you prepare plans and implement them?",
            "job": "12. Do you have a job?",
            "F15": "13. On a scale from 0-10, how content are you with your job?",
            "M8": "14. How would you rate your current finances?",
            "E17": "15. Who do you primarily talk to about personal and serious problems?",
            "G1": "16. Do you have a spouse/partner/boy-/girlfriend?",
            "J2": "17. How often have you spent time with other relatives in the past year?",
            "J4": "18. In the past year, how often have you spent time with acquaintances?",
            "J9": "19. In the past year, how often have you been to the cinema, a concert, or the theater?",
            "J17": "20. How often have you been abroad on holiday/family visits in the past year?"
        }

        for col, question in col_map.items():
            st.markdown(f"**{col}** → {question}")

    try:
        # Load SHAP & LIME objects
        shap_values = joblib.load("shap_values.pkl")
        explainer_shap = joblib.load("shap_explainer.pkl")
        lime_example = joblib.load("lime_example.pkl")

        import matplotlib.pyplot as plt
        import shap as shap_lib
        import numpy as np

        # === SHAP Section in dropdown ===
        with st.expander("📊 SHAP Explanation"):
            fig, ax = plt.subplots(figsize=(5, 3))
            shap_lib.summary_plot(
                shap_values, 
                features=None, 
                feature_names=X_columns, 
                show=False,
                max_display=10  # ✅ only top 10 important features
            )

            plt.tick_params(labelsize=7)   # smaller axis labels
            plt.title("Top 10 Features (SHAP)", fontsize=9)
            plt.xlabel("SHAP Value (impact on model output)", fontsize=8)

            st.pyplot(fig, clear_figure=True)

            st.markdown("""
            **What this means:**  
            The SHAP chart shows which features had the **biggest impact on the model’s prediction**.  
            - Bars further to the right mean **positive influence** (push prediction higher).  
            - Bars further to the left mean **negative influence** (push prediction lower).  
            - The top features listed are the most important factors overall.
            """)

        # === LIME Section in dropdown ===
        with st.expander("📊 LIME Explanation"):
            features, weights = zip(*lime_example)
            fig, ax = plt.subplots(figsize=(5, 3))
            y_pos = np.arange(len(features))
            ax.barh(y_pos, weights, align='center', color='skyblue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=7)
            ax.invert_yaxis()  # Most important on top
            ax.set_xlabel("Weight", fontsize=8)
            ax.set_title("LIME Explanation", fontsize=9)
            st.pyplot(fig, clear_figure=True)

            st.markdown("""
            **What this means:**  
            The LIME chart explains a **single prediction** by showing which features mattered most **for that user’s case**.  
            - Positive bars increased the predicted satisfaction level.  
            - Negative bars decreased it.  
            - Unlike SHAP (which looks at the model overall), LIME is **local** — it focuses on just one person’s prediction.
            """)

    except Exception as e:
        st.warning(f"Explainable AI output not available: {e}")





# Sidebar with additional features
with st.sidebar:
    st.header("🔧 Quick Actions")
    
    if st.button("🔄 Start New Assessment"):
        st.session_state.current_prediction = None
        st.session_state.current_confidence = None
        st.session_state.current_recommendations = None
        st.session_state.current_user_input = None
        st.session_state.chat_history = []
        st.rerun()
    
    if st.button("📋 View Assessment Guide"):
        st.info("""
        **Assessment Guide:**
        
        This assessment evaluates your life satisfaction across multiple dimensions:
        
        • **Health & Physical Wellbeing**
        • **Mental Health & Emotional Stability**
        • **Social Connections & Relationships**
        • **Work Satisfaction & Financial Security**
        • **Lifestyle & Recreational Activities**
        
        Be honest with your responses for the most accurate prediction!
        """)
    
    st.header("📞 Support")
    st.write("Need help? Use the AI chatbot or contact support.")
    
    st.header("🔒 Privacy")
    st.write("Your data is stored locally and never shared with third parties.")

# Footer
st.markdown("---")
st.markdown("🌱 *Life Satisfaction AI Assistant - Your personal guide to better wellbeing*")

