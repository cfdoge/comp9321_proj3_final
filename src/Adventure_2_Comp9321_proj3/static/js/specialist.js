$(document).ready(function() {
	console.log("lol");
	$( "#input_submit" ).on('click', function() {

		var suburb = document.getElementById('suburb_input').value;
		$("#suburb_input" ).val("");

		var unit_street = document.getElementById('unit_street_input').value;
		$("#unit_street_input" ).val("");

		var capacity = document.getElementById('capacity_input').value;
		$("#capacity_input" ).val("");

		var req = {};

		var location = {};
		// console.log(location);
		location["suburb"] = suburb;
		location["street_unit"] = unit_street;
		req["location"] = location;

		req["capacity"] = capacity;

		// console.log(req);
		const url = "http://127.0.0.1:5000/avg/rent/" + suburb;
		// console.log(url);
		$.getJSON(url,function(result){
			console.log(result);

			$('#house_price_modal').modal('show');
			var content= $('<div>'+'The average predicting price in'+result.loc +' is: $' + result.avg_rent +
                  '</div>');

			console.log(result.loc);
			console.log(result.avg_rent);
			console.log($('#avg_information').children().length);
			if($('#information').children().length == 0){
				$( "#information" ).append(content);
			}

			$('#house_price_modal').modal('show');
		});


	});
	
});