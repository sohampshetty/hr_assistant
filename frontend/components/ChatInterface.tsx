"use client"
import React, { useState, useRef, useEffect, KeyboardEvent, ChangeEvent } from "react";
import axios from "axios";
import styles from "../styles/ChatInterface.module.css";

type Message = {
  role: "user" | "assistant";
  content: string;
};

const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Typewriter effect: reveals text progressively
  const typewriterEffect = (text: string, updateCallback: (chunk: string) => void, delay = 30) => {
    let i = 0;
    const interval = setInterval(() => {
      i++;
      updateCallback(text.slice(0, i));
      if (i >= text.length) clearInterval(interval);
    }, delay);
  };

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage: Message = { role: "user", content: input.trim() };
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInput("");
    setLoading(true);

    try {
      const response = await axios.post("http://localhost:11434/v1/chat/completions", {
        model: "gemma3:1b",
        messages: updatedMessages,
        max_tokens: 200,
      });

      const fullText: string = response.data.choices[0].message.content.trim();

      // Start with empty assistant message
      setMessages([...updatedMessages, { role: "assistant", content: "" }]);

      // Reveal AI response progressively
      typewriterEffect(fullText, (chunk) => {
        setMessages((prevMessages) => {
          const msgs = [...prevMessages];
          // Update the last message content progressively
          msgs[msgs.length - 1] = { role: "assistant", content: chunk };
          return msgs;
        });
      });
    } catch (error) {
      setMessages((msg) => [
        ...msg,
        { role: "assistant", content: "Oops! Something went wrong while fetching the response." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    setInput(e.target.value);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className={styles.chatContainer} role="main" aria-label="HR Assistant Chat Interface">
      <h2 className={styles.headercss}>Chat with GenAI</h2>

      <div className={styles.messages} aria-live="polite" aria-atomic="false">
        {messages.length === 0 ? (
          <p className={styles.emptyPlaceholder}>
            Ask me about HR policies, leaves, or onboarding...
          </p>
        ) : (
          messages.map((msg, idx) => (
            <div
              key={idx}
              role="article"
              aria-label={msg.role === "user" ? "User message" : "Assistant message"}
              tabIndex={0}
              className={msg.role === "user" ? styles.messageUser : styles.messageAI}
            >
              {msg.content}
            </div>
          ))
        )}

        {loading && (
          <p aria-live="assertive" style={{ fontStyle: "italic", color: "#666" }}>
            AI is typing...
          </p>
        )}
        <div ref={bottomRef} />
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          sendMessage();
        }}
        className={styles.inputSection}
      >
        <input
          type="text"
          value={input}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          placeholder="Type your message here..."
          aria-label="Chat message input"
          disabled={loading}
          className={styles.inputBox}
        />
        <button type="submit" disabled={!input.trim() || loading} className={styles.sendButton} aria-label="Send chat message">
          {/* You can replace with a send icon here if you want */}
          Send
        </button>
      </form>
    </div>
  );
};

export default ChatInterface;
