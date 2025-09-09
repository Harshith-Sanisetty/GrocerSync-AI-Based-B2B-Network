// grocersync-backend/routes/users.js

const express = require('express');
const router = express.Router();
const { getSuppliers } = require('../controllers/userController');
const { protect } = require('../middleware/authMiddleware');

router.get('/suppliers', protect, getSuppliers);

module.exports = router;