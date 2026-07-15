import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Documents from './pages/Documents'
import LearningPath from './pages/LearningPath'
import Quiz from './pages/Quiz'
import Dashboard from './pages/Dashboard'
import Autonomous from './pages/Autonomous'
import Adaptive from './pages/Adaptive'

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/documents" element={<Documents />} />
        <Route path="/learning-path" element={<LearningPath />} />
        <Route path="/quiz" element={<Quiz />} />
        <Route path="/autonomous" element={<Autonomous />} />
        <Route path="/adaptive" element={<Adaptive />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="*" element={<Navigate to="/documents" replace />} />
      </Route>
    </Routes>
  )
}
