# 🌱 Life Satisfaction AI Assistant

A comprehensive AI-powered web application that predicts life satisfaction and provides personalized recommendations using machine learning and AI chatbot technology.

## ✨ Features

### 🤖 AI Chatbot Integration
- **Interactive AI Assistant**: Chat with an AI-powered assistant that explains your life satisfaction prediction
- **Personalized Guidance**: Get tailored advice based on your assessment results
- **Question Answering**: Ask questions about life satisfaction concepts and get instant responses
- **Assessment Support**: Receive guidance through the questionnaire process

### 📊 Real-Time Life Satisfaction Prediction
- **Instant Results**: Get immediate life satisfaction predictions without waiting
- **Confidence Scoring**: See how confident the model is in its prediction
- **Visual Analytics**: View probability distributions across different satisfaction levels
- **Comprehensive Assessment**: 20 carefully selected questions covering all life satisfaction dimensions

### 🎯 Personalized Recommendations
- **Actionable Advice**: Get specific recommendations for improving life satisfaction
- **Multi-Dimensional Focus**: Recommendations cover health, social, lifestyle, and professional areas
- **Prediction-Based**: Advice tailored to your specific life satisfaction level
- **Practical Steps**: Concrete actions you can take to improve your wellbeing

### 📈 Progress Tracking
- **Save Results**: Store your assessment results for future reference
- **Progress History**: View your previous assessments and track changes over time
- **Trend Analysis**: See how your life satisfaction evolves
- **Local Storage**: All data stored locally for privacy

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Required packages (see `req.txt`)

### Installation

1. **Install Dependencies**:
   ```bash
   pip install -r req.txt
   ```

2. **Run the Application**:
   ```bash
   streamlit run load_model_streamlit.py
   ```

3. **Access the App**:
   Open your browser and go to `http://localhost:8501`

## 📋 How to Use

### 1. Life Satisfaction Assessment
- Navigate to the "📊 Life Satisfaction Assessment" tab
- Fill out the comprehensive 20-question form
- Submit to get your instant prediction
- View confidence scores and probability distributions

### 2. AI Chatbot Assistant
- Go to the "🤖 AI Chatbot" tab
- Ask questions about your results or life satisfaction concepts
- Get personalized advice and explanations
- Use quick action buttons for common questions

### 3. Progress & Recommendations
- Check the "📈 Progress & Recommendations" tab
- View personalized recommendations based on your prediction
- Save your results for future tracking
- Review your assessment history

## 🔧 Technical Details

### AI Integration
- **DeepSeek API**: Powered by DeepSeek's advanced language model
- **Context-Aware**: Chatbot understands your specific prediction results
- **Real-Time Responses**: Instant AI-powered guidance and explanations

### Machine Learning Model
- **Trained Model**: Uses pre-trained life satisfaction prediction model
- **Feature Engineering**: Comprehensive feature encoding and scaling
- **Probability Estimation**: Provides confidence scores for predictions

### Data Privacy
- **Local Storage**: All user data stored locally
- **No External Sharing**: Your information never leaves your device
- **Secure Processing**: API calls only send necessary context

## 📊 Assessment Dimensions

The application evaluates life satisfaction across five key dimensions:

1. **Health & Physical Wellbeing**
   - General health rating
   - Physical health conditions
   - Medical expenses

2. **Mental Health & Emotional Stability**
   - Depression indicators
   - Loneliness assessment
   - Worry patterns
   - Emotional stability
   - Planning and implementation

3. **Work & Financial Security**
   - Employment status
   - Job satisfaction
   - Financial situation

4. **Social Connections & Relationships**
   - Support networks
   - Relationship status
   - Family connections
   - Social activities

5. **Lifestyle & Recreational Activities**
   - Cultural activities
   - Travel frequency
   - Personal characteristics

## 🎯 Recommendation Categories

Based on your life satisfaction level, you'll receive personalized recommendations in:

- **💪 Health & Wellness**: Physical and mental health strategies
- **👥 Social Connections**: Building and maintaining relationships
- **🌟 Lifestyle & Hobbies**: Personal development and recreation
- **💼 Professional Growth**: Career and skill development

## 🔒 Privacy & Security

- **Local Data Storage**: All assessment results stored locally
- **No Personal Data Collection**: No personally identifiable information collected
- **Secure API Communication**: Encrypted communication with AI services
- **User Control**: Complete control over your data and results

## 🛠️ Customization

### Adding New Questions
1. Update the `categories` dictionary in the code
2. Add new form fields in the Streamlit interface
3. Ensure the model supports the new features

### Modifying Recommendations
1. Edit the `get_personalized_recommendations()` function
2. Add new recommendation categories as needed
3. Customize advice based on your expertise

### AI Chatbot Customization
1. Modify the system prompt in the chatbot context
2. Add new quick action buttons
3. Customize response styles and lengths

## 📈 Future Enhancements

Potential improvements for future versions:

- **Mobile App**: Native mobile application
- **Advanced Analytics**: Detailed trend analysis and insights
- **Community Features**: Anonymous community sharing and support
- **Integration**: Connect with health apps and wearables
- **Multilingual Support**: Support for multiple languages
- **Advanced AI**: More sophisticated AI models and features

## 🤝 Contributing

This project is designed to be easily extensible. Feel free to:

- Add new assessment questions
- Improve the AI chatbot responses
- Enhance the recommendation system
- Add new visualization features
- Improve the user interface

## 📄 License

This project is for educational and research purposes. Please ensure compliance with all applicable regulations and ethical guidelines when using AI-powered assessment tools.

## 🆘 Support

For technical support or questions about the application:

1. Check the AI chatbot for immediate assistance
2. Review the assessment guide in the sidebar
3. Ensure all dependencies are properly installed
4. Verify your API key is correctly configured

---

**🌱 Life Satisfaction AI Assistant** - Empowering individuals to understand and improve their life satisfaction through AI-powered insights and personalized guidance.
