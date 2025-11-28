'use client'

import { createGlobalStyle } from 'styled-components'

const GlobalStyles = createGlobalStyle`
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  html {
    font-size: 16px;
    height: 100%;
    scroll-behavior: smooth;
  }

  body {
    font-family: 'Inter', sans-serif;
    height: 100%;
    padding: 0;
    margin: 0;
    background-color: #ffffff;
    color: #ffffff;
    line-height: 1.6;
  }

  a {
    color: inherit;
    text-decoration: none;
    transition: color 0.2s ease-in-out;
  }

  a:hover {
    color: #1f2937;
  }

  nav {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 60px;
    background-color: #ffffff;
    z-index: 9999;
    padding: 1rem 1.5rem;
    border-bottom: 1px solid #e5e7eb;
  }

  nav .container {
    max-width: 80rem;
    margin: 0 auto;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .logo-group {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .nav-menu {
      display: flex;
      align-items: center;
      margin-left: auto;
      gap: 0.25rem;
      font-size: 0.875rem;
      color: #374151;
  }


  .nav-menu a {
    padding: 0.5rem 0.75rem;
    transition: color 0.2s;
  }

  .nav-menu a:hover {
    color: #111827;
  }

  .nav-divider {
    color: #9ca3af;
    margin: 0 0.25rem;
  }

  main {
    height: 100%;
    background-color: #ffffff;
    padding: 0;
    max-width: 80rem;
    margin: 0 auto;
  }

  h1 {
    font-size: 1.5rem;
    font-weight: bold;
  }

  .logo-f { color: black; }
  .logo-h { color: #9333ea; }
  .logo-b { color: #2563eb; }
`

export default GlobalStyles
