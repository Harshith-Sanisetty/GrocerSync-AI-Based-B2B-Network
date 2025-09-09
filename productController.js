// grocersync-backend/controllers/productController.js

const Product = require('../models/Product');

// @desc    Get all products for a store owner
// @route   GET /api/products
// @access  Private
exports.getProducts = async (req, res) => {
  try {
    // Find products that belong to the logged-in user
    const products = await Product.find({ storeOwner: req.user.id });
    res.status(200).json(products);
  } catch (error) {
    res.status(500).json({ message: `Server Error: ${error.message}` });
  }
};

// @desc    Create a new product
// @route   POST /api/products
// @access  Private
exports.createProduct = async (req, res) => {
  try {
    const { name, sku, category, quantity, price, description } = req.body;

    const product = new Product({
      name,
      sku,
      category,
      quantity,
      price,
      description,
      storeOwner: req.user.id, // Assign the logged-in user as the owner
    });

    const createdProduct = await product.save();
    res.status(201).json(createdProduct);
  } catch (error) {
    res.status(500).json({ message: `Server Error: ${error.message}` });
  }
};

exports.updateProduct = async (req, res) => {
    try {
      const product = await Product.findById(req.params.id);
  
      if (!product) {
        return res.status(404).json({ message: 'Product not found' });
      }
  
      // Check if the logged-in user is the owner of the product
      if (product.storeOwner.toString() !== req.user.id) {
        return res.status(401).json({ message: 'User not authorized' });
      }
  
      const updatedProduct = await Product.findByIdAndUpdate(req.params.id, req.body, {
        new: true, // This option returns the modified document
      });
  
      res.status(200).json(updatedProduct);
    } catch (error) {
      res.status(500).json({ message: `Server Error: ${error.message}` });
    }
  };
  
  // @desc    Delete a product
  // @route   DELETE /api/products/:id
  // @access  Private
  exports.deleteProduct = async (req, res) => {
    try {
      const product = await Product.findById(req.params.id);
  
      if (!product) {
        return res.status(404).json({ message: 'Product not found' });
      }
  
      // Also check for ownership before deleting
      if (product.storeOwner.toString() !== req.user.id) {
        return res.status(401).json({ message: 'User not authorized' });
      }
  
      await Product.findByIdAndDelete(req.params.id);
  
      res.status(200).json({ message: 'Product removed successfully', id: req.params.id });
    } catch (error) {
      res.status(500).json({ message: `Server Error: ${error.message}` });
    }
  };
  