// grocersync-backend/routes/chat.js

const express = require('express');
const router = express.Router();
const { getMessages } = require('../controllers/chatController');
const { protect } = require('../middleware/authMiddleware');

router.get('/:receiverId', protect, getMessages);

module.exports = router;