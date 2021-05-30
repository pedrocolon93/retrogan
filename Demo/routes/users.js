var express = require('express');
var router = express.Router();

/* GET users listing. */
router.get('/', function(req, res, next) {
  res.render('layout',{title:'Network',content:'Potato'})
  // res.send('respond with a resource');
});

module.exports = router;
