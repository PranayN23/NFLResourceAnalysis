import { useState } from 'react'
import axios from 'axios'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  // Function to call the API when the button is clicked
  const handleUpload = async () => {
    try {
      // Call the backend API to trigger the upload action
      const response = await axios.post('http://127.0.0.1:5000/upload_data')
      
      // Log the response from the server
      console.log('Response from server:', response.data)
      alert('API called successfully!')
    } catch (error) {
      console.error('Error calling the API:', error)
      alert('Error calling the API.')
    }
  }

  return (
    <>
      <div>
        <a href="https://vite.dev" target="_blank">
          <img src={viteLogo} className="logo" alt="Vite logo" />
        </a>
        <a href="https://react.dev" target="_blank">
          <img src={reactLogo} className="logo react" alt="React logo" />
        </a>
      </div>
      <h1>Vite + React</h1>
      <div className="card">
        <button onClick={() => { setCount(count + 1); handleUpload() }}>
          count is {count}
        </button>
        <p>
          Edit <code>src/App.jsx</code> and save to test HMR
        </p>
      </div>
      <p className="read-the-docs">
        Click on the Vite and React logos to learn more
      </p>
    </>
  )
}

export default App
