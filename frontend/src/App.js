import './App.css';
import Navbar from './components/Navbar';
import Main from './components/Main';
import { useState } from 'react';
import Logs from './components/Logs';



function App() {
  const [selectedPage, setSelectedPage] = useState("lots");

  return (
    <section className='w-screen h-dvh overflow-hidden'>
      <Navbar selectedPage={selectedPage} setSelectedPage={setSelectedPage}/>
      {(selectedPage == "lots") ? 
      <Main/>
      :
      <Logs/>
      }
    </section>
  );
}

export default App;
