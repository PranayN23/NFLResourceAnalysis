import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { GlobalProvider } from './contexts/GlobalContext';
import Navbar from './Navbar';
import Home from './pages/Home';
import Login from './pages/Login';
import Signup from './pages/Signup';
import Results from './pages/Results';
<<<<<<< HEAD
import PlayerPage from './pages/PlayerPage';
import DraftPage from './pages/DraftPage'; // ✅ Import the new page
=======
import PlayerPage from './pages/PlayerPage'
import PlayerRanking from './pages/PlayerRanking';
import FreeAgency from './pages/FreeAgency'
>>>>>>> 9453fd46515d09d91ef3f4c0445758af9dc3b37d

import './App.css';

function App() {
  return (
    <GlobalProvider> {/* All components within this provider will be able to access GlobalContext via useContext. */}

      <BrowserRouter> {/* Within this, we can specify webapp routing. */}

        {/* Navbar must go inside browser router so the <Link> component works. */}
        <Navbar/>

        {/* Route each path to the corresponding component. */}
        <Routes>
          <Route path="/" element={<Home/>}/>
          <Route path="/login" element={<Login/>}/>
          <Route path="/signup" element={<Signup/>}/>
          <Route path="/results" element={<Results/>}/>
          <Route path="/player/:playerName" element={<PlayerPage />} />
<<<<<<< HEAD
          <Route path="/draft" element={<DraftPage />} /> {/* ✅ Add DraftPage route */}
=======
          <Route path="/player-rankings" element={<PlayerRanking />} />
          <Route path="/free-agency" element={<FreeAgency />} />
>>>>>>> 9453fd46515d09d91ef3f4c0445758af9dc3b37d
        </Routes>
      </BrowserRouter>

    </GlobalProvider>
  );
}

export default App;