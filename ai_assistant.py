import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date as dt_date, timedelta
import google.generativeai as genai
import streamlit as st
import plotly.express as px
import json
import uuid
import time
from financeAgent import FinanceAgent
from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')



# Streamlit UI
def main():
    st.set_page_config(
        page_title="AI Finance Agent",
        page_icon="💰",
        layout="wide"
    )
    
    if 'agent' not in st.session_state:
        st.session_state.agent = FinanceAgent()
        
    agent = st.session_state.agent
    
    st.title("💰 AI Finance Agent")
    st.markdown("Your intelligent financial assistant with memory and insights")
    
    # Sidebar for navigation and settings
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Choose a feature",
            ["Dashboard", "Transactions", "Analysis", "Budget & Goals", "Financial Advice", "Chat"]
        )
        
        st.markdown("---")
        
        # Demo data option
        if st.button("🧪 Add Demo Data"):
            agent.add_transaction(datetime(2025, 4, 1), -52.50, "Grocery shopping at Whole Foods")
            agent.add_transaction(datetime(2025, 4, 2), -125.30, "Uber rides for the week")
            agent.add_transaction(datetime(2025, 4, 3), 2500, "Monthly salary deposit")
            agent.add_transaction(datetime(2025, 4, 4), -85.20, "Dinner with friends at Italian restaurant")
            agent.add_transaction(datetime(2025, 4, 5), -45.99, "Netflix and Spotify subscriptions")
            agent.add_transaction(datetime(2025, 4, 6), -320.50, "New smartphone case and accessories")
            agent.add_transaction(datetime(2025, 4, 7), -960, "Monthly rent payment")
            agent.add_transaction(datetime(2025, 4, 8), -120.30, "Electricity and water bill")
            st.success("Demo data added!")
    
    # Main content area
    if page == "Dashboard":
        st.header("Financial Dashboard")
        
        # Top financial insights
        st.subheader("💡 Financial Insights")
        insights = agent.memory.get('agent_insights', [])
        if insights:
            for insight in insights[-3:]:
                st.info(insight)
        else:
            st.info("Add some transactions to get AI-powered insights!")
        
        # Quick stats
        col1, col2, col3 = st.columns(3)
        
        # Calculate basic metrics
        income = agent.df[agent.df['amount'] > 0]['amount'].sum()
        expenses = agent.df[agent.df['amount'] < 0]['amount'].sum() * -1
        balance = income - expenses
        
        with col1:
            st.metric("Total Income", f"${income:.2f}")
        with col2:
            st.metric("Total Expenses", f"${expenses:.2f}")
        with col3:
            st.metric("Balance", f"${balance:.2f}")
        
        # Recent transactions
        st.subheader("Recent Transactions")
        if not agent.df.empty:
            recent = agent.df.sort_values('date', ascending=False).head(5)
            st.dataframe(
                recent[['date', 'amount', 'description', 'category']],
                use_container_width=True
            )
        else:
            st.write("No transactions yet. Add some to get started!")
        
        # Category breakdown chart
        # Category breakdown chart
        st.subheader("Spending by Category")
        if not agent.df.empty and len(agent.df[agent.df['amount'] < 0]) > 0:
            expenses_df = agent.df[agent.df['amount'] < 0].copy()
            expenses_df['amount'] = expenses_df['amount'].abs()
            category_spending = expenses_df.groupby('category')['amount'].sum().reset_index()
            
            fig = px.pie(
                category_spending, 
                values='amount', 
                names='category',
                title='Expense Breakdown',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Add expense transactions to see category breakdown")
        
        # Budget progress
        st.subheader("Budget Progress")
        budget_goals = agent.memory.get('user_preferences', {}).get('budget_goals', {})
        if budget_goals:
            progress_df, narrative = agent.check_budget_progress()
            st.write(narrative)
            
            # Format progress as horizontal bars
            if not progress_df.empty:
                for _, row in progress_df.iterrows():
                    category = row['category']
                    percentage = min(100, row['percentage'])  # Cap at 100% for display
                    st.write(f"**{category}**: ${row['spent']:.2f} of ${row['goal']:.2f} ({percentage:.1f}%)")
                    st.progress(int(percentage))
        else:
            st.info("No budget goals set. Set them in the Budget & Goals section.")
            
        # Unusual transactions alert
        unusual_df, explanation = agent.detect_unusual_transactions()
        if unusual_df is not None and not unusual_df.empty:
            st.subheader("⚠️ Unusual Transaction Alert")
            st.warning(explanation)
            st.dataframe(unusual_df[['date', 'description', 'amount', 'category']])
    
    elif page == "Transactions":
        st.header("Manage Transactions")
        
        # Add new transaction
        st.subheader("Add New Transaction")
        col1, col2 = st.columns(2)
        
        with col1:
            date = st.date_input("Date", datetime.now())
            amount = st.number_input("Amount (negative for expenses)", step=0.01)
            
        with col2:
            description = st.text_input("Description")
            
            if st.button("Add Transaction", type="primary"):
                if description and amount != 0:
                    category, _ = agent.add_transaction(date, amount, description)
                    st.success(f"Transaction added and categorized as: {category}")
                else:
                    st.error("Please enter a description and non-zero amount")
        
        # Transaction history with filtering
        st.subheader("Transaction History")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_category = st.selectbox(
                "Filter by category", 
                ["All"] + list(agent.df['category'].unique() if not agent.df.empty else [])
            )
        with col2:
            filter_min_date = st.date_input(
                "From date", 
                datetime.now() - timedelta(days=30) if not agent.df.empty else datetime.now()
            )
        with col3:
            filter_max_date = st.date_input(
                "To date", 
                datetime.now()
            )
            
        # Apply filters
        filtered_df = agent.df.copy()
        if not filtered_df.empty:
            if filter_category != "All":
                filtered_df = filtered_df[filtered_df['category'] == filter_category]
                
            filtered_df = filtered_df[
                (filtered_df['date'].dt.date >= filter_min_date) & 
                (filtered_df['date'].dt.date <= filter_max_date)
            ]
        
        # Show filtered data
        if not filtered_df.empty:
            st.dataframe(
                filtered_df.sort_values('date', ascending=False)[['date', 'amount', 'description', 'category']],
                use_container_width=True
            )
            
            # Summary stats
            income = filtered_df[filtered_df['amount'] > 0]['amount'].sum()
            expenses = filtered_df[filtered_df['amount'] < 0]['amount'].sum() * -1
            balance = income - expenses
            
            st.markdown(f"""
            **Summary for selected period:**
            - Income: ${income:.2f}
            - Expenses: ${expenses:.2f}
            - Net: ${balance:.2f}
            """)
            
            # Download option
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Filtered Data",
                csv,
                "finance_data.csv",
                "text/csv",
                key='download-csv'
            )
        else:
            st.info("No transactions match your filters or no transactions added yet.")
            
        # Bulk import section
        with st.expander("Bulk Import"):
            st.write("Upload a CSV or Excel file with your transactions")
            st.markdown("""
            File must have these columns:
            - date (YYYY-MM-DD)
            - amount (positive for income, negative for expenses)
            - description
            
            Category will be determined automatically.
            """)
            
            uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        import_df = pd.read_csv(uploaded_file)
                    else:
                        import_df = pd.read_excel(uploaded_file)
                        
                    st.write("Preview:")
                    st.dataframe(import_df.head())
                    
                    if st.button("Import Data"):
                        # Basic validation
                        required_cols = ['date', 'amount', 'description']
                        if all(col in import_df.columns for col in required_cols):
                            for _, row in import_df.iterrows():
                                try:
                                    agent.add_transaction(
                                        pd.to_datetime(row['date']), 
                                        float(row['amount']),
                                        str(row['description'])
                                    )
                                except Exception as e:
                                    st.error(f"Error importing row: {e}")
                            
                            st.success(f"Successfully imported {len(import_df)} transactions!")
                        else:
                            st.error(f"File must contain these columns: {', '.join(required_cols)}")
                except Exception as e:
                    st.error(f"Error reading file: {e}")
    
    elif page == "Analysis":
        st.header("Financial Analysis")
        
        # Quick analysis buttons
        st.subheader("Quick Analysis")
        quick_queries = [
            "Show my spending by category in the last month",
            "How has my spending changed over time?",
            "What are my top 5 largest expenses?",
            "Show my income vs expenses by month",
            "What day of the week do I spend the most money?"
        ]
        
        selected_query = st.selectbox("Choose a quick analysis", quick_queries)
        custom_query = st.text_input("Or ask your own question", "")
        
        query = custom_query if custom_query else selected_query
        
        if st.button("Analyze"):
            with st.spinner("🧠 Analyzing your finances..."):
                result, fig, code = agent.analyze_data(query)
                
                # Display the result
                st.markdown("### Results")
                st.write(result)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                # Show the code for advanced users
                with st.expander("See the analysis code"):
                    st.code(code, language="python")
        
        # Advanced analysis options
        with st.expander("Advanced Analysis Options"):
            st.markdown("""
            Here are some examples of questions you can ask:
            
            - "What's my average daily spending on weekdays vs weekends?"
            - "Which category has grown the most in the last 3 months?"
            - "How much am I spending on subscription services?"
            - "Show my spending heatmap by day of week and time of day"
            - "What percentage of my income goes to essential vs non-essential expenses?"
            """)
    
    elif page == "Budget & Goals":
        st.header("Budget & Financial Goals")
        
        # Budget setting
        st.subheader("Set Budget Limits")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            categories = ["Food", "Transportation", "Housing", "Entertainment", 
                        "Shopping", "Utilities", "Healthcare", "Education", "Travel", "Other"]
            budget_category = st.selectbox("Category", categories)
        
        with col2:
            budget_amount = st.number_input("Monthly Budget Amount ($)", min_value=0.0, step=10.0)
            
        with col3:
            st.write("")
            st.write("")
            if st.button("Set Budget"):
                agent.set_budget_goal(budget_category, budget_amount)
                st.success(f"Budget for {budget_category} set to ${budget_amount:.2f}/month")
        
        # Current budgets
        budget_goals = agent.memory.get('user_preferences', {}).get('budget_goals', {})
        if budget_goals:
            st.subheader("Your Budget Goals")
            budget_df = pd.DataFrame([
                {"Category": cat, "Monthly Budget": f"${amount:.2f}"} 
                for cat, amount in budget_goals.items()
            ])
            st.dataframe(budget_df, use_container_width=True)
            
            # Check budget progress
            progress_df, narrative = agent.check_budget_progress()
            st.subheader("Current Month Progress")
            st.write(narrative)
            
            if not progress_df.empty:
                progress_formatted = progress_df.copy()
                progress_formatted['goal'] = progress_formatted['goal'].apply(lambda x: f"${x:.2f}")
                progress_formatted['spent'] = progress_formatted['spent'].apply(lambda x: f"${x:.2f}")
                progress_formatted['percentage'] = progress_formatted['percentage'].apply(lambda x: f"{x:.1f}%")
                
                st.dataframe(
                    progress_formatted.rename(columns={
                        'category': 'Category',
                        'goal': 'Budget',
                        'spent': 'Spent',
                        'percentage': 'Used',
                        'status': 'Status'
                    }),
                    use_container_width=True
                )
        else:
            st.info("No budget goals set yet. Create your first budget above.")
        
        # Savings goals
        st.markdown("---")
        st.subheader("Savings Goals")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            goal_amount = st.number_input("Goal Amount ($)", min_value=100.0, step=100.0)
            
        with col2:
            timeframe = st.number_input("Timeframe (months)", min_value=1, step=1)
            
        with col3:
            st.write("")
            st.write("")
            if st.button("Create Savings Plan"):
                with st.spinner("Creating your personalized savings plan..."):
                    saving_plan = agent.create_saving_plan(goal_amount, timeframe)
                    st.success("Savings plan created!")
                    st.markdown(saving_plan)
        
        # Budget recommendation
        with st.expander("Get Budget Recommendation"):
            if st.button("Generate Budget Recommendation"):
                with st.spinner("Creating personalized budget recommendation..."):
                    recommendation = agent.get_budget_recommendation()
                    st.markdown(recommendation)
    
    elif page == "Financial Advice":
        st.header("Financial Advice")
        
        # Personalized advice
        st.subheader("Get Personalized Advice")
        
        advice_query = st.text_input("What financial advice do you need?", 
                                    placeholder="e.g., How can I reduce my food expenses?")
        
        if st.button("Get Advice") or advice_query:
            with st.spinner("Generating personalized advice..."):
                advice = agent.get_financial_advice(advice_query)
                st.markdown(advice)
        
        # Pre-made advice topics
        st.subheader("Common Financial Topics")
        
        advice_topics = [
            "How to build an emergency fund",
            "Tips to reduce my monthly expenses",
            "How to improve my saving habits",
            "Smart ways to pay off debt faster",
            "How to start investing with small amounts"
        ]
        
        selected_topic = st.selectbox("Choose a topic", advice_topics)
        
        if st.button("Get Advice on this Topic"):
            with st.spinner("Generating advice..."):
                topic_advice = agent.get_financial_advice(selected_topic)
                st.markdown(topic_advice)
        
        # Financial health check
        with st.expander("Financial Health Check"):
            if st.button("Run Financial Health Check"):
                with st.spinner("Analyzing your financial health..."):
                    # This would be a more comprehensive analysis of the user's financial situation
                    prompt = """
                    You are a financial health analyst. Based on the user's transaction history,
                    perform a comprehensive financial health check covering:
                    
                    1. Income stability and consistency
                    2. Spending patterns and budget adherence
                    3. Saving rate and emergency fund status
                    4. Debt analysis (if applicable)
                    5. Overall financial health score (1-10)
                    6. Top 3 recommendations for improvement
                    
                    Format your response in clear sections with headers using markdown.
                    """
                    
                    response = model.generate_content(prompt)
                    health_check = response.text.strip()
                    st.markdown(health_check)
    
    elif page == "Chat":
        st.header("Chat with Your Finance Agent")
        
        # Initialize chat history
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
        
        # Display chat messages
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        user_input = st.chat_input("Ask me anything about your finances...")
        
        if user_input:
            # Add user message to chat history
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Get agent response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = agent.chat(user_input)
                    st.write(response)
            
            # Add assistant response to chat history
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
        
        # Suggestions for user
        if not st.session_state.chat_messages:
            st.info("👋 You can ask me things like:")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("- \"How much did I spend on food last month?\"")
                st.markdown("- \"What's my biggest expense category?\"")
                st.markdown("- \"Help me create a budget plan\"")
            with col2:
                st.markdown("- \"How can I save more money?\"")
                st.markdown("- \"Show me my spending trends\"")
                st.markdown("- \"What's my financial health like?\"")

if __name__ == "__main__":
    main()
