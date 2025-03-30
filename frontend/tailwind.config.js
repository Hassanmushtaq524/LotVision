 /** @type {import('tailwindcss').Config} */
 export default {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./src/components/**/*.{js,jsx,ts,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        'maroon': "#92007E",
        'reddish': "#92007E",
        'dark-gray': '#9E9D9E',
        'light-gray': '#9E9D9E'
      },
      fontFamily: {
        fontFamily: {
          'black': ['Inter-Black', 'Helvetica'],
          'light': ['Inter-Light', 'Helvetica'],
          'bold': ['Inter-Bold', 'Helvetica'],
          'thin': ['Inter-Thin', 'Helvetica']
        }
      }
    },
  },
  plugins: [],
}