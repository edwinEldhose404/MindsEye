import google.generativeai as genai
import os

def get_advice_from_gemini(api_key, emotion, diary_entry):
    try:
        # Configure the Gemini API with your key
        genai.configure(api_key=api_key)

        # Initialize the model
        # New line
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        # Create a detailed prompt for the model
        prompt = f"""
        Act as a compassionate, wise, and empathetic friend. Your goal is to provide supportive and actionable advice.

        A user is feeling: **{emotion}**

        Here is a snippet from their diary:
        ---
        "{diary_entry}"
        ---

        Based on their emotion and what they've written, please offer some gentle, constructive, and encouraging advice.
        Speak to them directly in a warm and understanding tone. If they seem to be in significant distress,
        gently suggest talking to a trusted person or a professional could be a good next step.
        """

        # Generate the content based on the prompt
        response = model.generate_content(prompt)

        return response.text

    except Exception as e:
        return f"An error occurred while generating advice: {e}"

def main():
    print("📝 Welcome to your AI-powered Diary Advisor!")
    print("Let's reflect on your entry.")

    api_key = os.getenv("GOOGLE_API_KEY")

    # Get user input
    print("-" * 30)
    user_emotion = input("How are you feeling right now? (e.g., happy, sad, anxious) ")
    user_diary_snippet = input("Please share a snippet from your diary for today: ")
    print("-" * 30)

    if not user_emotion or not user_diary_snippet:
        print("Both an emotion and a diary snippet are needed. Please try again.")
        return

    print("-" * 30)

    # Get and display the advice from Gemini
    advice = get_advice_from_gemini(api_key, user_emotion, user_diary_snippet)
    print(advice)
    print("-" * 30)


if __name__ == "__main__":
    main()