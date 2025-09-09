// grocersync-backend/routes/auth.js

const express = require('express');
const router = express.Router();

const { registerUser, loginUser, getMe } = require('../controllers/authController');
// Import the middleware
const { protect } = require('../middleware/authMiddleware');
// Define the registration route
router.post('/register', registerUser);



// 🔽 ADD THIS NEW ROUTE 🔽
router.post('/login', loginUser);
router.get('/me', protect, getMe);

module.exports = router;

