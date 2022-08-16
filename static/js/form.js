// $(document).ready(function () {
//   $('form').on('submit', function (event) {
//     var Generate = $('#Generate-Text').val();
//     // alert(Generate)
//     $.ajax({
//       // type: 'POST',
//       url: '/predict',
//       data: $('form').serialize(),
//       contentType: 'application/json',
//       success: function (data) {
//         console.log(data);
//         // alert("sucess" + data)
//       },
//       error: function (data) {
//         // alert("error " + data.error);
//       }
//     })
//       .done(function (data) {
//         if (data.error) {
//           $('#error-alert').text(data.error).show();
//         }
//         else {
//           $('#success-alert').text(data.Generate).show();
//         }
//       });

//     event.preventDefault();
//   });

// })


$('form').on('submit', function (event) {
  var text=$("#autocorrector").val()
  $.ajax({
    type: 'GET',
    url: '/prediction1/'+text,
    datatype:'json',
    crossdomain:true,
    success: function (data) {
      // Process on success
       var list = "";
        for(i=0; i<data[0].length; i++){
        list +="<li class='list-group-item d-flex justify-content-center'>"+data[0][i]+"</li>";
        }
        $("#suggestionslist").html(list);
        var list = "";
        for(i=0; i<data[1].length; i++){
        list +="<li class='list-group-item d-flex justify-content-center'>"+data[1][i]+"</li>";
        }
        $("#EditDistancelist").html(list);
        //$('#generic-sampling').text(data).show()
    },
    error: function (data) {
    }
  });

  event.preventDefault();
});