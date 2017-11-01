<?php

$host_name = 'db703376053.db.1and1.com';
$database = 'db703376053';
$user_name = 'dbo703376053';
$password = '';

//INSERT INTO `db703688159`.`markets_summaries` (`id`, `MarketName`, `High`, `Low`, `Volume`, `Last`, `BaseVolume`, `TimeStamp`, `Bid`, `Ask`, `OpenBuyOrders`, `OpenSellOrders`, `PrevDay`, `Created`, `added_at`) VALUES (NULL, '', '', '', '', '', '', NULL, '', '', '', '', '', '', CURRENT_TIMESTAMP);

$connect = mysqli_connect($host_name, $user_name, $password, $database);
if (mysqli_errno()) {
    die('<p>Failed to connect to MySQL: '.mysqli_error().'</p>');
} else {
	$url = "https://bittrex.com/api/v1.1/public/getmarketsummaries";

	$query = 'INSERT INTO `'.$database.'`.`markets_summaries` (`id`, `MarketName`, `High`, `Low`, `Volume`, `Last`, `BaseVolume`, `TimeStamp`, `Bid`, `Ask`, `OpenBuyOrders`, `OpenSellOrders`, `PrevDay`, `added_at`) VALUES ';
	$new = false;

	$json = json_decode(file_get_contents($url));
	
	foreach ($json->{'result'} as $json_row )
	{
    	$id = $json_row->{'MarketName'};
    	
    	$select_query = 'SELECT MAX(TimeStamp) as `TimeStamp` FROM `markets_summaries` WHERE `MarketName` = "'.$id.'";';
		$select_result = $connect->query($select_query);

    	$timestamp = $select_result->fetch_assoc()['TimeStamp'];
    	//echo '<div>'.$id.' : '.$timestamp.' : '.str_replace('T', ' ', $json_row->{'TimeStamp'}).'</div>';
    	//echo '<div>'.$id.' : '.abs(strtotime($timestamp) - strtotime(str_replace('T', ' ', $json_row->{'TimeStamp'}))).'</div>';
    	
    	// if more than 1 hour past get new data
    	if ($select_result->num_rows > 0 && (strtotime($json_row->{'TimeStamp'}) - strtotime($timestamp) < 3600))
    	{
        	continue;
        }
        // otherwise insert new row
        else
        {
	    	$new = true;
	    	$query .= ' (NULL, \''.$json_row->{'MarketName'}.'\', \''.$json_row->{'High'}.'\', \''.$json_row->{'Low'}.'\', \''.$json_row->{'Volume'}.'\', \''.$json_row->{'Last'}.'\', \''.$json_row->{'BaseVolume'}.'\', \''.str_replace('T', ' ', $json_row->{'TimeStamp'}).'\', \''.$json_row->{'Bid'}.'\', \''.$json_row->{'Ask'}.'\', \''.$json_row->{'OpenBuyOrders'}.'\', \''.$json_row->{'OpenSellOrders'}.'\', \''.$json_row->{'PrevDay'}.'\', CURRENT_TIMESTAMP), ';
        }
    	//echo '<div>'.$id.': '.$query.'</div>';
    	
    	//echo $connect->query($query);
	}

	$query = rtrim(rtrim($query,' '),',').';';
	if ($new )
	{
		//echo '<div>'.$query.'</div>';
		echo $connect->query($query);
	}

    mysqli_close($connect);
}
?>