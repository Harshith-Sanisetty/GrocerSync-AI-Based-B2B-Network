// grocersync-backend/models/User.js

const mongoose = require('mongoose');

// Define the schema for the User model
const userSchema = new mongoose.Schema(
  {
    name: {
      type: String,
      required: true,
      trim: true, // Removes whitespace from both ends
    },
    email: {
      type: String,
      required: true,
      unique: true, // Ensures every user has a unique email
      trim: true,
      lowercase: true, // Stores the email in lowercase
    },
    password: {
      type: String,
      required: true,
    },
    role: {
      type: String,
      enum: ['store_owner', 'supplier'], // The role must be one of these values
      required: true,
    },
    storeName: { // Only relevant if the role is 'store_owner'
      type: String,
      trim: true,
    }
  },
  {
    // Automatically adds 'createdAt' and 'updatedAt' fields
    timestamps: true, 
  }
);

// Create the User model from the schema and export it
const User = mongoose.model('User', userSchema);

module.exports = User;