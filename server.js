// server.js

const express = require('express');
const dotenv = require('dotenv');
const cors = require('cors');
const mongoose = require('mongoose');
const http = require('http');
const { Server } = require("socket.io");

dotenv.config();

// --- Database Connection ---
const connectDB = async () => {
  try {
    await mongoose.connect(process.env.MONGO_URI);
    console.log('✅ MongoDB connected successfully!');
  } catch (error) {
    console.error(`❌ MongoDB connection error: ${error.message}`);
    process.exit(1);
  }
};

connectDB();

// --- Express App Setup ---
const app = express();
const PORT = process.env.PORT || 5001;

// --- Import Routes ---
const authRoutes = require('./routes/auth');
const productRoutes = require('./routes/products.js');
const userRoutes = require('./routes/users.js');
const chatRoutes = require('./routes/chat.js'); 

// --- Middleware ---
app.use(cors());
app.use(express.json());

// --- Use Routes ---
app.use('/api/auth', authRoutes);
app.use('/api/products', productRoutes);
app.use('/api/users', userRoutes);
app.use('/api/chat', chatRoutes);

// Test route
app.get('/', (req, res) => {
  res.status(200).json({ message: 'Welcome to the GrocerSync API!' });
});

// --- Socket.io Integration ---
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

// Use the external chat handler
require('./socket/chatHandler')(io);

// --- Start Server ---
// ✅ This should be the ONLY server.listen call and it should be at the end.
server.listen(PORT, () => {
  console.log(`🚀 Server is running on port ${PORT}`);
});