import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { GlobalProvider } from './contexts/GlobalContext';
import Navbar from './Navbar';
import Home from './pages/Home';
import Login from './pages/Login';
import Signup from './pages/Signup';
import Results from './pages/Results';
import PlayerPage from './pages/PlayerPage';
import PlayerRanking from './pages/PlayerRanking';
import TeamPffRanking from './pages/TeamPffRanking';
import DraftPage from './pages/DraftPage'; // ✅ Import the new page
import FreeAgency from './pages/FreeAgency'

import './App.css';

function App() {
  return (
    <GlobalProvider>
      <BrowserRouter>
        <Navbar />

        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/results" element={<Results />} />
          <Route path="/player/:playerName" element={<PlayerPage />} />
          <Route path="/draft" element={<DraftPage />} /> {/* ✅ Add DraftPage route */}
          <Route path="/player-rankings" element={<PlayerRanking />} />
          <Route path="/team-pff" element={<TeamPffRanking />} />
          <Route path="/free-agency" element={<FreeAgency />} />
        </Routes>
      </BrowserRouter>
    </GlobalProvider>
  );
}

export default App;