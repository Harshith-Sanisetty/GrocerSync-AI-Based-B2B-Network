// grocersync-backend/controllers/chatController.js

const ChatMessage = require('../models/ChatMessage');

// @desc    Get messages between two users
// @route   GET /api/chat/:receiverId
// @access  Private
exports.getMessages = async (req, res) => {
  try {
    const { receiverId } = req.params;
    const senderId = req.user.id;

    // Find all messages where the sender/receiver pair matches, sorted by time
    const messages = await ChatMessage.find({
      $or: [
        { sender: senderId, receiver: receiverId },
        { sender: receiverId, receiver: senderId },
      ],
    }).sort({ createdAt: 'asc' });

    res.json(messages);
  } catch (error) {
    res.status(500).json({ message: 'Server Error' });
  }
};