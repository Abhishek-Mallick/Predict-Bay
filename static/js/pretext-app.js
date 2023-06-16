const form = document.getElementById("chat-form");
const input = document.getElementById("chat-input");
const messages = document.getElementById("chat-messages");
const apiKey = "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";
// deepraj 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

document.getElementById("chat-icon").addEventListener("click", () => {
    const chatContainer = document.getElementById("chat-container");
    chatContainer.style.display = chatContainer.style.display === "none" ? "block" : "none";
});

const preText =
    "I am your personal stock market analyst, here to provide you with comprehensive information about stocks and companies in a well formatted manner with numbers. I can help you with details such as years on the market, market cap, volatility, market sector, liquid assets, treasury, and any other information a stock buyer should know. Please provide me with a stock ticker and ask any questions related to the stock market. If your question is not related to stocks, I'll kindly remind you to ask questions related to stocks, and I'll answer you to the best of my knowledge.When you greet me with a hello, I'll greet you back with [Hi! I am Dan your personal stock market analyst.] ";

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const message = input.value;
    input.value = "";

    messages.innerHTML += `<div class="message user-message">
  <img src="static/img/chatbot_2.jpg" alt="user icon"> <span>${message}</span>
  </div>`;

    // Use axios library to make a POST request to the OpenAI API
    const response = await axios.post(
        "https://api.openai.com/v1/completions",
        {
            prompt: preText + `${message}`,
            model: "text-davinci-003",
            temperature: 0.9,
            max_tokens: 500,
            top_p: 1,
            frequency_penalty: 0.0,
            presence_penalty: 0.6,
        },
        {
            headers: {
                "Content-Type": "application/json",
                Authorization: `Bearer ${apiKey}`,
            },
        }
    );
    const chatbotResponse = response.data.choices[0].text;

    messages.innerHTML += `<div class="message bot-message">
  <img src="static/img/chatbot_1.png" alt="bot icon"> <span>${chatbotResponse}</span>
  </div>`;
});