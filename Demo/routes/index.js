var express = require('express');
var vis = require('vis');
var router = express.Router();

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index', { title: 'Post-Specialization Visualizer' });
});

module.exports = router;
