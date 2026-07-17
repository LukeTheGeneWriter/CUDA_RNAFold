$(document).ready(function(){

  /*
   *  work-around the nasty 'container' class that gets automatically
   *  added to each 'container' by sphinx since it collides with
   *  bootstraps 'container' class
   */
  $(".brief-description")
    .removeClass("container");

  $(".detailed-description")
    .removeClass("container");

  /*
   *  wrap all symbol containers in a bootstrap panel
   */
  $(".macro-container")
    .removeClass("container")
    .addClass("panel-body")
    .wrap("<div class='panel panel-default'></div>");

  $(".typedef-container")
    .removeClass("container")
    .addClass("panel-body")
    .wrap("<div class='panel panel-default'></div>");

  $(".function-container")
    .removeClass("container")
    .addClass("panel-body")
    .wrap("<div class='panel panel-default'></div>");

  $(".fields-container")
    .removeClass("container")
    .addClass("panel-body")
    .wrap("<div class='panel panel-default'></div>");

  $(".variables-container")
    .removeClass("container")
    .addClass("panel-body")
    .wrap("<div class='panel panel-default'></div>");

  $(".enum-container")
    .removeClass("container")
    .addClass("panel-body")
    .wrap("<div class='panel panel-default'></div>");
});
