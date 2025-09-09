// grocersync-backend/models/Product.js

const mongoose = require('mongoose');

const productSchema = new mongoose.Schema(
  {
    name: {
      type: String,
      required: true,
      trim: true,
    },
    sku: { // Stock Keeping Unit
      type: String,
      required: true,
      unique: true,
    },
    category: {
      type: String,
      required: true,
      trim: true,
    },
    quantity: {
      type: Number,
      required: true,
      default: 0,
    },
    price: {
      type: Number,
      required: true,
    },
    description: {
      type: String,
    },
    // Creates a link to the User model
    storeOwner: {
      type: mongoose.Schema.Types.ObjectId,
      required: true,
      ref: 'User', 
    },
  },
  {
    timestamps: true,
  }
);

const Product = mongoose.model('Product', productSchema);

module.exports = Product;