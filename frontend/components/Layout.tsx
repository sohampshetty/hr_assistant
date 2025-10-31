import React, { ReactNode } from "react";
import styles from "../styles/Layout.module.css";

interface LayoutProps {
  children: ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <div className={styles.container}>
      <aside className={styles.sidebar}>
        <h2 className={styles.sidebarTitle}>Chats</h2>
        <div className={styles.sidebarContent}>
          <p>No chats yet.</p>
        </div>
      </aside>

      <div className={styles.main}>
        <header className={styles.header}>HR Assistant</header>
        <main className={styles.contentArea}>{children}</main>
      </div>
    </div>
  );
};

export default Layout;
