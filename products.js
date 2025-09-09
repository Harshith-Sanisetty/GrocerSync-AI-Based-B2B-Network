// grocersync-backend/routes/products.js

const express = require('express');
const router = express.Router();
const { 
    getProducts, 
    createProduct,
    updateProduct,
    deleteProduct
  } = require('../controllers/productController');
const { protect } = require('../middleware/authMiddleware');

// Apply the 'protect' middleware to both routes
router.route('/')
  .get(protect, getProducts)
  .post(protect, createProduct);
  
router.route('/:id')
  .put(protect, updateProduct)
  .delete(protect, deleteProduct);
module.exports = router;