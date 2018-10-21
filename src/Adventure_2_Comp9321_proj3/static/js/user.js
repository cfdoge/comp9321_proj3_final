$(document).ready(function() {

	$( "#input_submit" ).on('click', function() {

		var suburb = document.getElementById('suburb_input').value;
		$("#suburb_input" ).val("");

		var unit_street = document.getElementById('unit_street_input').value;
		$("#unit_street_input" ).val("");


		var capacity = document.getElementById('capacity_input').value;
		$("#capacity_input" ).val("");

		var room_type = document.getElementById('type_input').value;
		$("#type_input" ).val("");

		var price = document.getElementById('price_input').value;
		$("#price_input" ).val("");


		var req = {};
		var location = {};
		// console.log(location);
		location["suburb"] = suburb;
		location["street_unit"] = unit_street;
		req["location"] = location;
		req["capacity"] = capacity
		req["room_type"] = room_type;
		req["price"] = price;

		console.log(req);
		var xhr = new XMLHttpRequest();
		var url = "http://127.0.0.1:5000/roomsearch";
		xhr.open("post", url, true);
		xhr.setRequestHeader("Content-Type", "application/json");

		var data = JSON.stringify(req);
		xhr.send(data);
		$('#information').empty();


	})

	$("#get_submit" ).on('click', function() {

		var url = "http://127.0.0.1:5000/roomsearch";
		$.getJSON(url,function(result){
			// prediction_price_json = JSON.parse(result)
			// prediction_price = result.price;
			console.log(Object.keys(result).length);

			var i;
            var keys = Object.keys(result);
			for (i = 0; i < Object.keys(result).length; i++) {
//
				var p = result[keys[i]]['price']
				var n = result[keys[i]]['name']
			    var content= $('<div>'+'The name is: $' + p+ ', name: '+ n +
                  '</div>');
			    $( "#information" ).append(content);
			}



			$('#house_price_modal').modal('show');
		});


	});

});