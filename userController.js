// grocersync-backend/controllers/userController.js

const User = require('../models/User');

// @desc    Get all users with the role 'supplier'
// @route   GET /api/users/suppliers
// @access  Private
exports.getSuppliers = async (req, res) => {
  try {
    // Find all users where the role is 'supplier'
    const suppliers = await User.find({ role: 'supplier' }).select('-password');
    res.json(suppliers);
  } catch (error) {
    res.status(500).json({ message: 'Server Error' });
  }
};