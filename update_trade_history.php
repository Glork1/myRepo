<?php
$host_name = 'db703688159.db.1and1.com';
$database = 'db703688159';
$user_name = 'dbo703688159';
$password = '';

$select_query = 'SELECT `id`, `ccy` FROM `tickers`;';

$connect = mysqli_connect($host_name, $user_name, $password, $database);
if (mysqli_errno()) {
    mail('nika.shishonkova@gmail.com','update_trade_history on Poloniex: Failed to connect to MySQL: '.mysqli_error(),phpversion());
} else {
	$select_result = $connect->query($select_query);
	
	if ($select_result->num_rows > 0)
	{
		// for each ticker
		while($row = $select_result->fetch_assoc()) 
		{
			$max_id_query = "SELECT MAX(id_ext) as `max` FROM `trade_history` WHERE ccy = '".$row['id']."';";
	        $max_id_result = $connect->query($max_id_query);
	        $max_id = ($max_id_result->num_rows > 0 ? $max_id_result->fetch_assoc()['max'] : 1e16);
			
			//echo "<div>".$row['id']." : ".$max_id." - ".time()."</div>";
			
			$end_timestamp = time();
			$start_timestamp = $end_timestamp - 7200;
							
			//echo "<div>".$start_timestamp." - ".$end_timestamp."</div>";
			
			$insert_query = 'INSERT INTO `'.$database.'`.`trade_history` (`id`, `id_ext`, `ccy`, `tradeID`, `TimeStamp`, `Quantity`, `Price`, `Total`, `FillType`, `type`, `added_at`) VALUES ';
				
			$url = "https://poloniex.com/public?command=returnTradeHistory&currencyPair=".$row['ccy']."&start=".$start_timestamp."&end=".$end_timestamp;
			
			//echo "<div>".$url."</div>";
			
        	$json = json_decode(file_get_contents($url));
        	$query = '';
        	
    		foreach ($json as $json_row)
			{
				$id = $json_row->{'globalTradeID'};
	        	if ($id <= $max_id) 
	        	{
		        	continue;
		        }
				
		    	$insert_line = "(NULL, '".$json_row->{'globalTradeID'}."', '".$row['id']."', '".$json_row->{'tradeID'}."', '".$json_row->{'date'}."', '".$json_row->{'amount'}."', '".$json_row->{'rate'}."', '".$json_row->{'total'}."', NULL, '".$json_row->{'type'}."', CURRENT_TIMESTAMP), ";

	        	$insert_query .= $insert_line;
			}
			
			$insert_query = rtrim(rtrim($insert_query,' '),',').';';
			//echo '<div>'.$insert_query.'</div>';
			echo $connect->query($insert_query);
			
			$start_timestamp = $end_timestamp + 1;
			sleep(2);
		}
	}
	
	//echo $connect->query($query);

    mysqli_close($connect);
}
?>