// server.js

const express = require('express');
const dotenv = require('dotenv');
const cors = require('cors');
const mongoose = require('mongoose');
const http = require('http');
const { Server } = require("socket.io");

dotenv.config();


const connectDB = async () => {
  try {
    await mongoose.connect(process.env.MONGO_URI);
    console.log(' MongoDB connected successfully!');
  } catch (error) {
    console.error(` MongoDB connection error: ${error.message}`);
    process.exit(1);
  }
};

connectDB();

const app = express();
const PORT = process.env.PORT || 5001;


const authRoutes = require('./routes/auth');
const productRoutes = require('./routes/products.js');
const userRoutes = require('./routes/users.js');
const chatRoutes = require('./routes/chat.js'); 


app.use(cors());
app.use(express.json());


app.use('/api/auth', authRoutes);
app.use('/api/products', productRoutes);
app.use('/api/users', userRoutes);
app.use('/api/chat', chatRoutes);


app.get('/', (req, res) => {
  res.status(200).json({ message: 'Welcome to the GrocerSync API!' });
});


const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});


require('./socket/chatHandler')(io);


server.listen(PORT, () => {
  console.log(`🚀 Server is running on port ${PORT}`);

});
