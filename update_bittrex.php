<?php

$host_name = 'db703376053.db.1and1.com';
$database = 'db703376053';
$user_name = 'dbo703376053';
$password = '';

$select_query = 'SELECT `MarketName` FROM `markets` WHERE IsActive = 1;';



$connect = mysqli_connect($host_name, $user_name, $password, $database);
if (mysqli_errno()) {
    die('<p>Failed to connect to MySQL: '.mysql_error().'</p>');
} else {
	
	echo '<div>'.date("Y-m-d g:i:s").'</div>';
	
	$insert_query = 'INSERT INTO `db703376053`.`mkt_history` (`id`, `id_ext`, `ccy`, `TimeStamp`, `Quantity`, `Price`, `Total`, `FillType`, `OrderType`, `added_at`) VALUES ';
	$select_result = $connect->query($select_query);
	
	if ($select_result->num_rows > 0)
	{
		while($row = $select_result->fetch_assoc()) {
			$ccy = $row["MarketName"];
			
        	$url = "https://bittrex.com/api/v1.1/public/getmarkethistory?market=".$ccy;
        	
        	$max_id_query = "SELECT MAX(id_ext) as `max` FROM `mkt_history` WHERE ccy = '".$ccy."';";
        	$max_id_result = $connect->query($max_id_query);
        	
        	$max_id = ($max_id_result->num_rows > 0 ? $max_id_result->fetch_assoc()['max'] : 1e16);
        	
        	$json = json_decode(file_get_contents($url));
        	
        	foreach ($json->{'result'} as $json_row )
        	{
	        	$id = $json_row->{'Id'};
	        	if ($id <= $max_id) 
	        	{
		        	continue;
		        }
	        	
	        	$insert_line = "(NULL, '".$json_row->{'Id'}."', '".$row["MarketName"]."', '".str_replace('T', ' ', $json_row->{'TimeStamp'})."', '".$json_row->{'Quantity'}."', '".$json_row->{'Price'}."', '".$json_row->{'Total'}."', '".$json_row->{'FillType'}."', '".$json_row->{'OrderType'}."', CURRENT_TIMESTAMP), ";

	        	$insert_query .= $insert_line;
        	}
    	}
	}
	
	$insert_query = rtrim(rtrim($insert_query,' '),',').';';
	
	echo $connect->query($insert_query);
    
    mysqli_close($connect);
}
?>
